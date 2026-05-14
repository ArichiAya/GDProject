import os
import pandas as pd
from nfstream import NFStreamer


def run_extraction(pcap_path, output_csv):
    """
    使用 NFStream 提取特征，加入抗 Flood 强杀机制
    """
    if not os.path.exists(pcap_path):
        return 0

    # 保护机制：防止处理超大文件导致内存溢出
    file_size_mb = os.path.getsize(pcap_path) / (1024 * 1024)
    if file_size_mb > 100:
        print(f"[!] 文件过大 ({file_size_mb:.2f}MB)，已跳过以保护系统。")
        return 0

    try:
        # idle_timeout=10: 针对无响应的洪水包，10秒没动作直接结算
        # active_timeout=15: 强制流的最长生存周期，保证实时性
        streamer = NFStreamer(source=pcap_path,
                              statistical_analysis=True,
                              idle_timeout=10,
                              active_timeout=15)

        df = streamer.to_pandas()
        if df is None or df.empty:
            return 0

        df.to_csv(output_csv, index=False)
        print(f"[+] 提取完成: {len(df)} 条原始流特征。")
        return len(df)

    except Exception as e:
        print(f"[!] 提取失败: {e}")
        return 0
