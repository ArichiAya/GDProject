import os
from nfstream import NFStreamer


def run_extraction():
    # 建议使用原始字符串或正斜杠处理 Windows 路径
    pcap_path = r"D:\GraduationProject\src\evil2.pcap"

    print(f"正在检查文件是否存在: {pcap_path}")

    if not os.path.exists(pcap_path):
        print("错误：找不到 pcap 文件！")
        return

    # 初始化提取器
    # n_dissection=0 可以关闭深度包检测，进一步加快速度
    streamer = NFStreamer(source=pcap_path, statistical_analysis=True)

    # 转换为 Pandas DataFrame
    print("正在提取流量特征，请稍候...")
    df = streamer.to_pandas()

    # 打印前几行确认
    print(f"提取成功！共发现 {len(df)} 条流。")
    print(df.head())

    # 保存结果
    df.to_csv("evil2.csv", index=False)


# --- 关键修复：防止递归启动子进程 ---
if __name__ == "__main__":
    run_extraction()
