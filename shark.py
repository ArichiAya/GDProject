import time
import os
import sys
from scapy.all import sniff, wrpcap

# --- 配置 ---
TARGET_IFACE = "VMware Virtual Ethernet Adapter for VMnet8"
TARGET_IP = "192.168.80.128"
SAVE_DIR = "uploads"
FILTER_STR = f"tcp and host {TARGET_IP}"
TOTAL_CYCLE = 30


def capture_tcp_loop():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"[*] 监控启动...")

    while True:
        current_time = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{SAVE_DIR}/tcp_only_{current_time}.pcap"
        print(f"[*] 开启新周期文件: {file_name}")

        # 定义一个内部函数来处理每个包
        def packet_handler(pkt):
            # 直接追加写入文件，不占用内存
            wrpcap(file_name, pkt, append=True)

        try:
            # 捕获指定时间段的流量
            # store=0 表示不保存在内存中，这是防止 OOM（内存溢出）的关键
            sniff(iface=TARGET_IFACE,
                  filter=FILTER_STR,
                  timeout=TOTAL_CYCLE,
                  prn=packet_handler,
                  store=0)

            if os.path.exists(file_name):
                print(f"[+] 周期结束，文件已保存。")
            else:
                print(f"[-] 周期内无符合条件的流量。")

        except KeyboardInterrupt:
            print("\n[!] 用户中断，正在退出...")
            return
        except Exception as e:
            print(f"[!] 错误: {e}")
            time.sleep(2)


if __name__ == "__main__":
    capture_tcp_loop()
