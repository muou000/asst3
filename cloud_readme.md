# AWS 配置指南 (AWS Setup Instructions)

[English Version / 英文原版](cloud_readme_en.md)

为了进行性能测试，你需要在亚马逊云服务（AWS）上的虚拟机（VM）实例上运行此作业。我们将提供（或已经发送）学生优惠券供你用于计费。以下是在 AWS 上设置和运行环境的详细步骤。

> [!NOTE]
> 请切记：当天使用完毕后务必**关闭（SHUT DOWN / STOP）**你的实例，以避免因整夜运行而耗尽免费额度！

## 创建虚拟机 (Creating a VM)

1. 登录 [AWS EC2 控制台 (AWS EC2 dashboard)](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Home)。在页面右上角，请确保你所在的区域与你申请配额的区域一致（通常随机选自 `us-west-2`、`ap-northeast-1`、`ap-northeast-2`）。
![Switch Region](handout/switch-region.png?raw=true)

2. 现在可以创建虚拟机实例了。点击 **`Launch instance`（启动实例）** 按钮。
![Launch instance](handout/launch-instance.png?raw=true)

3. 点击 **`Browse more AMIs`（浏览更多 AMI）**。
![Search AMI](handout/search-ami.png)

4. 在 **`Community AMIs`（社区 AMI）** 标签页中搜索：`Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250613`。
![alt text](handout/select-ami.jpeg)

5. 选择 **`g5g.xlarge`** 实例类型。
![Instance type](handout/instance-type.png?raw=true)

6. 将磁盘卷（Volume）大小修改为 **65 GiB**，以容纳作业所需的各种依赖软件包：
![Storage](handout/storage.jpeg?raw=true)

7. 你需要一个密钥对（Key pair）来访问你的实例。在 **`Key pair (login)`（密钥对）** 部分，点击 **`Create a new key pair`（创建新密钥对）**，可以任意命名。这会将一个名为 `<key_name>.pem` 的密钥文件下载到你的电脑上，后续你将使用它登录即将创建的 VM 实例。最后即可启动实例。
![Key Pair Step 1](handout/keypair-step1.png)
![Key Pair Step 2](handout/keypair-step2.png)

8. 确认所有配置详情无误后，点击启动实例（Launch instance）：
![Confirm](handout/confirm-launch.jpeg)

9. 虚拟机创建完成后，你就可以通过 **SSH** 登录了。你需要实例的公网 IPv4 DNS 名称（Public IPv4 DNS name），可以通过导航到实例页面，点击 **`Connect`（连接）** 按钮，然后选择 **`SSH`** 选项卡找到（注意：实例可能需要一点时间启动并分配 IP 地址）：
![Connect](handout/connect.png?raw=true)

请确保按照提示修改密钥文件的权限，在本地终端运行：
```bash
chmod 400 path/to/key_name.pem
```

获取公网 DNS / IP 地址后，可以使用以下命令登录实例：
```bash
ssh -i path/to/key_name.pem ubuntu@<public_dns_name>
```

> [!WARNING]
> 创建实例后，如果在配置期间需要离开，请务必将其关机停止运行。保持运行会持续消耗你的额度，并可能产生额外费用。

## 配置虚拟机环境 (Setting up the VM environment)

1. 默认情况下系统已经安装了 CUDA。你可以使用 `nvidia-smi` 命令检查 CUDA 版本，输出应该类似于以下内容。CUDA 版本应为 **12.8**，使用的 GPU 是 **NVIDIA T4G**：
```text
$ nvidia-smi
Thu Oct 23 18:22:32 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   53C    P8             13W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

2. 运行以下命令安装本作业所需的 FreeGlut 库：
```bash
sudo apt-get update && sudo apt-get install freeglut3-dev -y
```

3. 现在你可以克隆 asst3 git 仓库并开始开发了：
```bash
git clone https://github.com/stanford-cs149/asst3.git
```

## 从 AWS 提取/同步代码 (Fetching your code from AWS)

我们建议你创建一个私有 git 仓库（如 GitHub / GitLab 私有仓库），在里面进行作业开发。这样可以降低丢失代码的风险，并方便跟踪历史版本。

或者，你也可以在本地机器上使用 `scp` 命令从远程机器提取文件：
```bash
scp -i <path-to-your-private-key> lightsail-user@<instance-IP-addr>:/path/to/file /path/to/local_file
```

## 关闭虚拟机 (Shutting down VM)

当你结束使用 VM 时，可以通过在 AWS 网页端点击“停止实例（stop computer / Stop instance）”来关机，或者在终端中运行以下命令：
```bash
sudo shutdown -h now
```
