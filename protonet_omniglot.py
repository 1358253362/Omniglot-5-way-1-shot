"""Omniglot 5-way 1-shot: ProtoNet 完整可运行实现。

流程
----
1. 一次性把 32,460 张图缩放到 28x28 并缓存到内存（约 100MB），避免每个
   episode 重复读盘；
2. 按类别划分 train / val / test（背景字母表训练，评估字母表测试）；
3. 每个 episode 采样 N 个类、每类 k 个 support + q 个 query；
4. 用 support 的类均值作为原型，query 按最近原型分类，交叉熵训练；
5. 测试集上采样数百个 episode，报告平均准确率与 95% 置信区间。

用法
----
    python protonet_omniglot.py
    python protonet_omniglot.py --train-episodes 5000 --test-shots 1,5,10
    python protonet_omniglot.py --device cpu --train-episodes 500 --no-cache

数据
----
首次运行调用 torchvision 下载；若 GitHub 直连失败，自动改从 jsDelivr 镜像
下载并解压到 <root>/omniglot-py/。
"""

import argparse
import hashlib
import os
import random
import socket
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Omniglot

MD5 = {
    "images_background": "68d2efa1b9178cc56df9314c21c6e718",
    "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
}

MIRRORS = (
    "https://cdn.jsdelivr.net/gh/brendenlake/omniglot@master/python/{}.zip",
    "https://ghproxy.net/https://github.com/brendenlake/omniglot/raw/master/python/{}.zip",
    "https://ghfast.top/https://github.com/brendenlake/omniglot/raw/master/python/{}.zip",
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/{}.zip",
)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_fallback(root):
    """torchvision 下载失败时，从镜像下载并解压。

    torchvision 用 <root>/omniglot-py/<subset>.zip 的 MD5 判断数据是否完整，
    因此 zip 必须放在 omniglot-py/ 里，而不是 <root>/ 下。
    """
    target = os.path.join(root, "omniglot-py")
    os.makedirs(target, exist_ok=True)
    socket.setdefaulttimeout(180)

    for name in ("images_background", "images_evaluation"):
        zip_path = os.path.join(target, name + ".zip")
        if not (os.path.exists(zip_path) and _md5(zip_path) == MD5[name]):
            last_err = None
            for tpl in MIRRORS:
                url = tpl.format(name)
                tmp = zip_path + ".part"
                try:
                    print("[data] 下载 " + name + ".zip：" + url.split("/")[2])
                    urllib.request.urlretrieve(url, tmp)
                    if _md5(tmp) != MD5[name]:
                        raise RuntimeError("MD5 校验失败")
                    os.replace(tmp, zip_path)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    print("[data] 该镜像失败：" + type(exc).__name__)
            if last_err is not None:
                raise RuntimeError(
                    "无法下载 " + name + ".zip，请手动下载后放到 " + zip_path
                ) from last_err
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
    print("[data] 数据就绪：" + target)


def build_index(root):
    """返回 {类别id: [图片路径, ...]} 与背景集类别数。"""
    try:
        bg = Omniglot(root=root, background=True, download=True)
        ev = Omniglot(root=root, background=False, download=True)
    except Exception as exc:
        print("[data] 官方下载失败（" + type(exc).__name__ + "），切换镜像")
        download_fallback(root)
        bg = Omniglot(root=root, background=True, download=False)
        ev = Omniglot(root=root, background=False, download=False)

    def collect(ds, offset):
        out = []
        for cls_idx, character in enumerate(ds._characters):
            for fn, label in ds._character_images[cls_idx]:
                path = os.path.join(ds.target_folder, character, fn)
                out.append((path, label + offset))
        return out

    bg_pairs = collect(bg, 0)
    offset = max(label for _, label in bg_pairs) + 1
    ev_pairs = collect(ev, offset)

    class_to_paths = {}
    for path, label in bg_pairs + ev_pairs:
        class_to_paths.setdefault(label, []).append(path)
    return class_to_paths, offset


def split_classes(class_to_paths, offset, seed=0):
    """背景字母表：864 类训练 + 100 类验证；评估字母表：659 类测试。"""
    bg_classes = [c for c in class_to_paths if c < offset]
    ev_classes = [c for c in class_to_paths if c >= offset]
    rng = random.Random(seed)
    rng.shuffle(bg_classes)
    return bg_classes[:864], bg_classes[864:964], ev_classes


def build_cache(class_to_paths, size=28, workers=8):
    """预读所有图片并缩放到 28x28，返回 {类别: [tensor, ...]}。"""
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def load(path):
        return tfm(Image.open(path).convert("L"))

    cache, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (cls, paths) in enumerate(class_to_paths.items(), 1):
            cache[cls] = list(ex.map(load, paths))
            if i % 400 == 0:
                print("[cache] %d/%d classes (%.1fs)" % (i, len(class_to_paths), time.time() - t0))
    print("[cache] done: %d classes, %.1fs" % (len(cache), time.time() - t0))
    return cache


def make_episode(cache, classes, n_way, k_shot, q_query, rng, aug=None):
    """采样一个 episode：N 个类，每类 k 个 support + q 个 query。"""
    chosen = rng.sample(classes, n_way)
    support, support_y, query, query_y = [], [], [], []
    for label, cls in enumerate(chosen):
        images = cache[cls]
        picked = rng.sample(range(len(images)), k_shot + q_query)
        for j, idx in enumerate(picked):
            img = images[idx]
            if aug is not None:
                img = aug(img)
            if j < k_shot:
                support.append(img)
                support_y.append(label)
            else:
                query.append(img)
                query_y.append(label)
    return (torch.stack(support), torch.tensor(support_y),
            torch.stack(query), torch.tensor(query_y))


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ProtoNet(nn.Module):
    """4 层卷积，28x28 -> 1x1，输出 64 维嵌入。"""

    def __init__(self, width=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(1, width),
            conv_block(width, width),
            conv_block(width, width),
            conv_block(width, width),
        )

    def forward(self, x):
        return self.encoder(x).flatten(1)


def prototypes(features, labels, n_way):
    return torch.stack([features[labels == c].mean(0) for c in range(n_way)])


def evaluate(model, cache, classes, n_way, k_shot, q_query, episodes, device, seed=1234):
    model.eval()
    rng = random.Random(seed)
    accs = []
    with torch.no_grad():
        for _ in range(episodes):
            s, sy, q, qy = make_episode(cache, classes, n_way, k_shot, q_query, rng)
            s, sy, q, qy = s.to(device), sy.to(device), q.to(device), qy.to(device)
            proto = prototypes(model(s), sy, n_way)
            logits = -torch.cdist(model(q), proto).pow(2)
            accs.append((logits.argmax(1) == qy).float().mean().item())
    accs = torch.tensor(accs)
    mean = accs.mean().item()
    ci = 1.96 * accs.std(unbiased=True).item() / len(accs) ** 0.5
    return mean, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=15)
    parser.add_argument("--test-shots", default="1,5")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", default="protonet_omniglot.pt")
    parser.add_argument("--no-cache", action="store_true", help="关闭内存缓存（慢，但省内存）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("device=" + device)

    class_to_paths, offset = build_index(args.root)
    train_classes, val_classes, test_classes = split_classes(class_to_paths, offset, args.seed)
    n_images = sum(len(v) for v in class_to_paths.values())
    print("classes: total=%d train=%d val=%d test=%d images=%d"
          % (len(class_to_paths), len(train_classes), len(val_classes), len(test_classes), n_images))

    if args.no_cache:
        tfm = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        cache = {c: [tfm(Image.open(p).convert("L")) for p in ps] for c, ps in class_to_paths.items()}
    else:
        cache = build_cache(class_to_paths, workers=args.workers)

    aug = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))

    model = ProtoNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.train_episodes // 3), gamma=0.5)

    rng = random.Random(args.seed + 1)
    model.train()
    t0 = time.time()
    running_loss, running_acc = [], []
    for ep in range(1, args.train_episodes + 1):
        s, sy, q, qy = make_episode(cache, train_classes, args.n_way, args.k_shot,
                                    args.q_query, rng, aug)
        s, sy, q, qy = s.to(device), sy.to(device), q.to(device), qy.to(device)

        proto = prototypes(model(s), sy, args.n_way)
        logits = -torch.cdist(model(q), proto).pow(2)
        loss = F.cross_entropy(logits, qy)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        running_loss.append(loss.item())
        running_acc.append((logits.argmax(1) == qy).float().mean().item())
        if ep % 100 == 0 or ep == args.train_episodes:
            print("ep %5d loss %.4f train-acc %.4f (%.1fs)"
                  % (ep, sum(running_loss) / len(running_loss),
                     sum(running_acc) / len(running_acc), time.time() - t0))
            running_loss, running_acc = [], []

    torch.save(model.state_dict(), args.save)
    print("saved -> " + args.save)

    val_acc, val_ci = evaluate(model, cache, val_classes, args.n_way, args.k_shot,
                               args.q_query, args.eval_episodes, device)
    print("val  5-way %d-shot: %.2f%% +/- %.2f" % (args.k_shot, val_acc * 100, val_ci * 100))

    for shots in [int(x) for x in args.test_shots.split(",") if x.strip()]:
        acc, ci = evaluate(model, cache, test_classes, args.n_way, shots,
                           args.q_query, args.eval_episodes, device)
        print("test 5-way %d-shot: %.2f%% +/- %.2f  (%d episodes)"
              % (shots, acc * 100, ci * 100, args.eval_episodes))


if __name__ == "__main__":
    main()
