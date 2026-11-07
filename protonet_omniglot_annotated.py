"""Omniglot 5-way 1-shot：ProtoNet 注释版。
k-kay : support set 里有 k 类
n-shot : 每类里有 n 个样本

与 protonet_omniglot.py 的算法完全一致，只加了中文注释。
建议读法：
  1. 先看 main() 里的五段：数据 -> 划分 -> 缓存 -> 训练 -> 评测；
  2. 再看 make_episode() 和 ProtoNet：这两块是方法本体；
  3. 剩下的（镜像下载、缓存、命令行）都是工程细节。
"""

# ---- 标准库 ----
import argparse          # 解析命令行参数
import hashlib           # 计算 zip 的 MD5，和 torchvision 的校验值对齐
import os                # 路径拼接、文件存在性判断
import random            # episode 采样、类别打乱（Python 自带，便于复现）
import socket            # 设置全局网络超时，避免下载卡死
import time              # 计时
import urllib.request    # 从镜像下载 zip
import zipfile           # 解压 zip
from concurrent.futures import ThreadPoolExecutor  # 多线程读图（I/O 密集）

# ---- 第三方库 ----
import torch                     # 张量与自动求导
import torch.nn as nn            # 网络层
import torch.nn.functional as F  # 交叉熵等函数式接口
from PIL import Image            # 读 PNG
from torchvision import transforms          # 缩放、随机仿射
from torchvision.datasets import Omniglot   # 官方数据加载器

# ---- 可视化：无显示环境下也能保存图片 ----
import matplotlib
matplotlib.use("Agg")            # 必须在 import pyplot 之前设置
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 一、数据下载：torchvision 失败时的兜底
# ---------------------------------------------------------------------------

# 两个子集的 MD5，必须与 torchvision.datasets.Omniglot.zips_md5 完全一致，
# 否则即使文件下好了，torchvision 仍会判定“数据损坏”。
MD5 = {
    "images_background": "68d2efa1b9178cc56df9314c21c6e718",
    "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
}

# 依次尝试的镜像：先 jsDelivr（通常最快），再走 GitHub 代理，最后官方源。
MIRRORS = (
    "https://cdn.jsdelivr.net/gh/brendenlake/omniglot@master/python/{}.zip",
    "https://ghproxy.net/https://github.com/brendenlake/omniglot/raw/master/python/{}.zip",
    "https://ghfast.top/https://github.com/brendenlake/omniglot/raw/master/python/{}.zip",
    "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/{}.zip",
)


def _md5(path):
    """分块计算文件 MD5，避免把几十 MB 一次性读进内存。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 每次 1MB
            h.update(chunk)
    return h.hexdigest()


def download_fallback(root):
    """torchvision 下载失败时，从镜像下载并解压。

    注意目录层级：torchvision 校验的是 <root>/omniglot-py/<subset>.zip 的
    MD5，所以 zip 必须落在 omniglot-py/ 里，不能放在 <root>/ 下。
    """
    target = os.path.join(root, "omniglot-py")  # torchvision 只认这个子目录名
    os.makedirs(target, exist_ok=True)          # 已存在时不报错
    socket.setdefaulttimeout(180)               # 单次网络操作最多等 3 分钟

    for name in ("images_background", "images_evaluation"):
        zip_path = os.path.join(target, name + ".zip")

        # 已存在且 MD5 正确 -> 直接用，不重复下载
        if not (os.path.exists(zip_path) and _md5(zip_path) == MD5[name]):
            last_err = None
            for tpl in MIRRORS:                 # 逐个镜像试
                url = tpl.format(name)
                tmp = zip_path + ".part"        # 先写临时文件，避免半截包被误认
                try:
                    print("[data] 下载 " + name + ".zip：" + url.split("/")[2])
                    urllib.request.urlretrieve(url, tmp)
                    if _md5(tmp) != MD5[name]:  # 内容不对，直接丢弃
                        raise RuntimeError("MD5 校验失败")
                    os.replace(tmp, zip_path)   # 校验通过才改名，原子操作
                    last_err = None
                    break
                except Exception as exc:        # 该镜像失败，换下一个
                    last_err = exc
                    print("[data] 该镜像失败：" + type(exc).__name__)
            if last_err is not None:            # 所有镜像都失败，给出明确路径
                raise RuntimeError(
                    "无法下载 " + name + ".zip，请手动下载后放到 " + zip_path
                ) from last_err

        with zipfile.ZipFile(zip_path) as zf:   # 解压出 images_xxx/ 目录
            zf.extractall(target)
    print("[data] 数据就绪：" + target)


# ---------------------------------------------------------------------------
# 二、建立索引：把 1623 个类摊平成“类别 -> 图片路径列表”
# ---------------------------------------------------------------------------

def build_index(root):
    """返回 {类别id: [图片路径, ...]} 与背景集类别数。"""
    try:
        # 正常路径：官方源下载 + MD5 校验，失败会抛异常
        bg = Omniglot(root=root, background=True, download=True)
        ev = Omniglot(root=root, background=False, download=True)
    except Exception as exc:
        # 网络失败（SSL、超时）时走镜像，然后用 download=False 重新加载
        print("[data] 官方下载失败（" + type(exc).__name__ + "），切换镜像")
        download_fallback(root)
        bg = Omniglot(root=root, background=True, download=False)
        ev = Omniglot(root=root, background=False, download=False)

    def collect(ds, offset):
        """把一个子集摊平成 [(绝对路径, 类别编号)]。"""
        out = []
        for cls_idx, character in enumerate(ds._characters):
            # _characters[cls_idx] 是该字符类所在目录，如 "Angelic/..."
            for fn, label in ds._character_images[cls_idx]:
                # torchvision 只存文件名，必须自己拼出完整路径
                path = os.path.join(ds.target_folder, character, fn)
                out.append((path, label + offset))
        return out

    bg_pairs = collect(bg, 0)                          # 背景集编号从 0 开始
    offset = max(label for _, label in bg_pairs) + 1   # 评估集从这里往后排
    ev_pairs = collect(ev, offset)                     # 两套编号不重叠

    class_to_paths = {}
    for path, label in bg_pairs + ev_pairs:
        class_to_paths.setdefault(label, []).append(path)  # 按类聚合成列表
    return class_to_paths, offset


# ---------------------------------------------------------------------------
# 三、划分：背景字母表训练/验证，评估字母表测试
# ---------------------------------------------------------------------------

def split_classes(class_to_paths, offset, seed=0):
    """背景字母表：864 类训练 + 100 类验证；评估字母表：659 类测试。"""
    bg_classes = [c for c in class_to_paths if c < offset]   # 964 个背景类
    ev_classes = [c for c in class_to_paths if c >= offset]  # 659 个评估类

    rng = random.Random(seed)   # 固定种子，保证每次划分一致
    rng.shuffle(bg_classes)     # 打乱后再切，避免按字母表顺序造成偏置

    # 前 864 训练、接着 100 验证；评估字母表全部作测试（测试类训练时不可见）
    return bg_classes[:864], bg_classes[864:964], ev_classes


# ---------------------------------------------------------------------------
# 四、内存缓存：把 32,460 张图一次性缩放到 28x28
# ---------------------------------------------------------------------------

def build_cache(class_to_paths, size=28, workers=8):
    """预读所有图片并缩放到 28x28，返回 {类别: [tensor, ...]}。"""
    tfm = transforms.Compose([
        transforms.Resize((size, size)),  # 统一尺寸
        transforms.ToTensor(),            # PIL(HWC,0-255) -> CHW float,0-1
    ])

    def load(path):
        return tfm(Image.open(path).convert("L"))  # 转灰度单通道

    cache, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (cls, paths) in enumerate(class_to_paths.items(), 1):
            # ex.map 保持顺序，每类仍是固定的 20 张
            cache[cls] = list(ex.map(load, paths))
            if i % 400 == 0:  # 每 400 类报一次进度
                print("[cache] %d/%d classes (%.1fs)" % (i, len(class_to_paths), time.time() - t0))
    print("[cache] done: %d classes, %.1fs" % (len(cache), time.time() - t0))
    return cache  # 约 32,460 × 784 × 4B ≈ 100MB


# ---------------------------------------------------------------------------
# 五、采样 episode：这个方法的核心
# ---------------------------------------------------------------------------
def make_episode(cache, classes, n_way, k_shot, q_query, rng, aug=None):
    """采样一个 episode：N 个类，每类 k 个 support + q 个 query。"""
    chosen = rng.sample(classes, n_way)  # 无放回抽 N 个类
    support, support_y, query, query_y = [], [], [], []

    for label, cls in enumerate(chosen):
        # label 是 0..N-1 的临时编号，每个 episode 重新开始
        images = cache[cls] #len(images)=20
        # 一次抽 k+q 张，保证同一张图不会既当 support 又当 query
        picked = rng.sample(range(len(images)), k_shot + q_query)

        for j, idx in enumerate(picked):
            img = images[idx]
            if aug is not None:   # 训练时做随机仿射；评测时传 None
                img = aug(img)
            if j < k_shot:        # 前 k 张 -> support
                support.append(img)
                support_y.append(label)
            else:                 # 其余 -> query
                query.append(img)
                query_y.append(label)

    # 堆成 (N*K,1,28,28) / (N*Q,1,28,28)，标签转 long 供交叉熵使用
    return (torch.stack(support), torch.tensor(support_y),
            torch.stack(query), torch.tensor(query_y))


# ---------------------------------------------------------------------------
# 六、网络：4 层卷积，把 28x28 压成 64 维
# ---------------------------------------------------------------------------

def conv_block(in_c, out_c):
    """一个卷积块：卷积 -> BN -> ReLU -> 池化，尺寸减半。"""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),  # padding=1 保持长宽不变
        nn.BatchNorm2d(out_c),                 # 稳定训练，允许更大学习率
        nn.ReLU(inplace=True),                 # 非线性
        nn.MaxPool2d(2),                       # 28->14->7->3->1
    )


class ProtoNet(nn.Module):
    """4 层卷积，28x28 -> 1x1，输出 64 维嵌入。"""

    def __init__(self, width=64):
        super().__init__()  # 必须先初始化父类，否则参数注册不上
        self.encoder = nn.Sequential(
            conv_block(1, width),        # 1 通道灰度图 -> 64
            conv_block(width, width),    # 64 -> 64
            conv_block(width, width),    # 64 -> 64
            conv_block(width, width),    # 64 -> 64，输出 1x1
        )
    """model(x)调用方法"""
    def forward(self, x):
        # (B,64,1,1) -> (B,64)，后面直接比距离，不再接全连接
        return self.encoder(x).flatten(1)


# ---------------------------------------------------------------------------
# 七、原型与评测
# ---------------------------------------------------------------------------

def prototypes(features, labels, n_way):
    """对每个类取 support 特征的均值，得到 (N,64) 的原型矩阵。"""
    # labels == c 是布尔掩码，mean(0) 沿样本维求平均
    # labels也是[0,n_way)范围
    return torch.stack([features[labels == c].mean(0) for c in range(n_way)])


def evaluate(model, cache, classes, n_way, k_shot, q_query, episodes, device, seed=1234):
    """在给定类别池上采样若干 episode，返回平均准确率与 95% 置信区间。"""
    model.eval()  # 关键：BN 改用滑动统计量，否则 1-shot 时统计量噪声极大
    rng = random.Random(seed)  # 固定测试 episode，保证结果可复现
    accs = []

    with torch.no_grad():  # 评测不需要梯度
        for _ in range(episodes):
            s, sy, q, qy = make_episode(cache, classes, n_way, k_shot, q_query, rng)
            s, sy, q, qy = s.to(device), sy.to(device), q.to(device), qy.to(device)

            proto = prototypes(model(s), sy, n_way)          # (N,64)
            # 负的平方欧氏距离：距离越小，logit 越大
            logits = -torch.cdist(model(q), proto).pow(2)    # (Q,N)
            accs.append((logits.argmax(1) == qy).float().mean().item())

    accs = torch.tensor(accs)
    mean = accs.mean().item()
    # 95% CI = 1.96 * 标准差 / sqrt(episode 数)
    ci = 1.96 * accs.std(unbiased=True).item() / len(accs) ** 0.5
    return mean, ci


# ---------------------------------------------------------------------------
# 七点五、训练曲线可视化
# ---------------------------------------------------------------------------

def plot_training_curves(loss_history, acc_history, val_history, save_path):
    """画训练曲线，x 轴为任务数（episode）。

    loss_history / acc_history: [(ep, value)] 每个 episode 一个点
    val_history:               [(ep, value)] 每 100 个 episode 一次
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- 左图：训练损失 ----
    if loss_history:
        eps, losses = zip(*loss_history)
        axes[0].plot(eps, losses, linewidth=1.0, label="train loss")
    axes[0].set_xlabel("episode (task)")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ---- 右图：训练 / 验证准确率 ----
    if acc_history:
        eps, accs = zip(*acc_history)
        axes[1].plot(eps, accs, linewidth=1.0, label="train acc")
    if val_history:
        eps, accs = zip(*val_history)
        # 验证点稀疏，用带标记的线突显
        axes[1].plot(eps, accs, marker="o", linewidth=1.6, label="val acc")
    axes[1].set_xlabel("episode (task)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print("[plot] curves saved -> " + save_path)


# ---------------------------------------------------------------------------
# 八、主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")                 # 数据目录
    parser.add_argument("--train-episodes", type=int, default=3000)  # 训练 episode 数
    parser.add_argument("--eval-episodes", type=int, default=600)    # 评测 episode 数
    parser.add_argument("--n-way", type=int, default=5)              # 测试时每次几个类
    parser.add_argument("--k-shot", type=int, default=1)             # 每类几张 support
    parser.add_argument("--q-query", type=int, default=15)           # 每类几张 query
    parser.add_argument("--test-shots", default="1,5")               # 用同一模型测多个 shot
    parser.add_argument("--lr", type=float, default=1e-3)            # 学习率
    parser.add_argument("--seed", type=int, default=0)               # 随机种子
    parser.add_argument("--device", default="auto")                  # auto/cuda/cpu
    parser.add_argument("--save", default="protonet_omniglot.pt")    # 权重保存路径
    parser.add_argument("--plot", default="protonet_curves.png")     # 训练曲线保存路径
    parser.add_argument("--no-cache", action="store_true", help="关闭内存缓存（慢，但省内存）")           # 关闭内存缓存
    parser.add_argument("--workers", type=int, default=8)            # 读图线程数
    args = parser.parse_args()
    train_n_way = 60  # 训练时每次几个类

    random.seed(args.seed)
    torch.manual_seed(args.seed)  # 固定随机性，保证可复现

    # 设备选择：auto 时有 GPU 就用 GPU
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("device=" + device)

    # --- 1. 数据 ---
    class_to_paths, offset = build_index(args.root)
    train_classes, val_classes, test_classes = split_classes(class_to_paths, offset, args.seed)
    n_images = sum(len(v) for v in class_to_paths.values())
    print("classes: total=%d train=%d val=%d test=%d images=%d"
          % (len(class_to_paths), len(train_classes), len(val_classes), len(test_classes), n_images))

    # --- 2. 缓存 ---
    if args.no_cache:  # 省内存模式：仍会全部读一遍，只是不常驻
        tfm = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        cache = {c: [tfm(Image.open(p).convert("L")) for p in ps] for c, ps in class_to_paths.items()}
    else:
        cache = build_cache(class_to_paths, workers=args.workers)

    # 训练增强：±10° 旋转、10% 平移、0.9~1.1 缩放
    aug = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))

    # --- 3. 模型与优化器 ---
    model = ProtoNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 每 train_episodes/3 步学习率减半，共降两次
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.train_episodes // 3), gamma=0.5)

    # --- 训练曲线记录容器（x 轴 = 任务数 episode）---
    loss_history = []   # [(ep, train_loss)]  每个 episode 一个点
    acc_history = []    # [(ep, train_acc)]   每个 episode 一个点
    val_history = []    # [(ep, val_acc)]     每 100 个 episode 一个点

    # --- 4. 训练：每个 episode 更新一次参数 ---
    rng = random.Random(args.seed + 1)  # 与划分用不同的随机流
    model.train()
    t0 = time.time()
    running_loss, running_acc = [], []
    best_val = -1.0
    best_state = None
    bad = 0
    patience = 5
    min_delta = 0.002

    # train_episodes训练任务
    for ep in range(1, args.train_episodes + 1):
        s, sy, q, qy = make_episode(cache, train_classes, train_n_way, args.k_shot,
                                    args.q_query, rng, aug)
        s, sy, q, qy = s.to(device), sy.to(device), q.to(device), qy.to(device)

        """---弄懂这一块做了什么，就可以写ppt了---"""
        proto = prototypes(model(s), sy, train_n_way)     # 用 support 算原型(找到每一类的特征原型（模型）)
        # 把“查询样本到各类原型的距离”变成“分类 logits”，距离越小分数越高。
        logits = -torch.cdist(model(q), proto).pow(2)    # query 与原型比距离:(75, 64) (5，64)
        loss = F.cross_entropy(logits, qy)               # 计算交叉熵损失（隐含 softmax）
        """-----------------"""
        # s和q什么关系：对应位置样本属于同一类

        opt.zero_grad()   # 清空旧梯度
        loss.backward()   # 反向传播，计算梯度
        opt.step()        # 用梯度更新参数
        sched.step()      # 更新学习率调度器

        # 记录当前任务的 loss 与训练准确率（用于画曲线）
        loss_history.append((ep, loss.item()))
        acc_history.append((ep, (logits.argmax(1) == qy).float().mean().item()))

        # 累计 loss 和训练准确率，仅用于打印
        running_loss.append(loss.item())
        running_acc.append((logits.argmax(1) == qy).float().mean().item())

        if ep % 100 == 0 or ep == args.train_episodes:
            print("ep %5d loss %.4f train-acc %.4f (%.1fs)"
                  % (ep, sum(running_loss) / len(running_loss),
                     sum(running_acc) / len(running_acc), time.time() - t0))
            running_loss, running_acc = [], []
            val_acc, val_ci = evaluate(model, cache, val_classes, args.n_way, args.k_shot,
                                           args.q_query, args.eval_episodes, device)
            model.train()  # evaluate 里调了 model.eval()
            print("ep %d val %.2f%% +/- %.2f" % (ep, val_acc * 100, val_ci * 100))
            val_history.append((ep, val_acc))   # 记录验证点

            if val_acc > best_val + min_delta:
                best_val = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("early stop at ep %d, best val %.2f%%" % (ep, best_val * 100))
                    break

    # --- 训练结束（含早停），无条件保存训练曲线 ---
    plot_training_curves(loss_history, acc_history, val_history, args.plot)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), args.save)  # 只存权重，不存模型对象
        print("saved -> " + args.save)

        # 同一个模型，换不同 shot 数再测，观察“多给样本值多少”
        for shots in [int(x) for x in args.test_shots.split(",") if x.strip()]:
            acc, ci = evaluate(model, cache, test_classes, args.n_way, shots,
                               args.q_query, args.eval_episodes, device)
            print("test 5-way %d-shot: %.2f%% +/- %.2f  (%d episodes)"
                  % (shots, acc * 100, ci * 100, args.eval_episodes))

        for shots in [int(x) for x in args.test_shots.split(",") if x.strip()]:
            acc, ci = evaluate(model, cache, test_classes, 20, shots,
                                       args.q_query, args.eval_episodes, device)
            print("test 20-way %d-shot: %.2f%% +/- %.2f  (%d episodes)"
                          % (shots, acc * 100, ci * 100, args.eval_episodes))

if __name__ == "__main__":
    main()  # 被 import 时不执行