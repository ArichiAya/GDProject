import os
from nfstream import NFStreamer


def run_extraction(pcap_path, output_csv):
    """
    接收前端传入的 pcap 路径，并将提取的特征保存为指定的 csv 路径
    """
    print(f"正在处理上传的文件: {pcap_path}")

    if not os.path.exists(pcap_path):
        raise FileNotFoundError("错误：找不到 pcap 文件！")

    # 初始化提取器
    streamer = NFStreamer(source=pcap_path, statistical_analysis=True)

    # 转换为 Pandas DataFrame
    df = streamer.to_pandas()

    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"特征提取成功！共发现 {len(df)} 条流，已保存至 {output_csv}。")
    return len(df)


if __name__ == "__main__":
    # 本地测试用的默认路径
    run_extraction(r"D:\GraduationProject\src\evil2.pcap", "evil2.csv")
