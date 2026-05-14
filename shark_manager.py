# 文件名：shark_manager.py
import time
import os
import threading
from scapy.all import sniff
from scapy.utils import PcapWriter

# --- 配置 ---
TARGET_IFACE = "VMware Virtual Ethernet Adapter for VMnet8"
# TARGET_IP = "192.168.80.128"
SAVE_DIR = "traffic_data"
FILTER_STR = "tcp and src host 192.168.80.128 and dst host 192.168.80.1"


class SlidingWindowSniffer:

    def __init__(self, callback=None):
        self.is_running = False
        self.stop_event = threading.Event()
        self.window_size = 15  # 每个窗口抓取 15 秒
        self.step_size = 15  # 每 15 秒启动一个新窗口
        self.callback = callback  # 每 15 秒启动一个新窗口 (首尾相接，不重叠)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_event.clear()
        # 启动主控线程
        threading.Thread(target=self._master_loop, daemon=True).start()
        print("[*] 滑动窗口流量监控已启动...")

    def stop(self):
        self.is_running = False
        self.stop_event.set()
        print("[!] 正在停止监控，等待当前窗口安全关闭...")

    def _master_loop(self):
        """主控循环：每隔 step_size 启动一个新的线程"""
        while not self.stop_event.is_set():
            threading.Thread(target=self._sniff_worker, daemon=True).start()
            # 挂起 15 秒，响应停止事件会立即打破挂起
            self.stop_event.wait(self.step_size)

    def _sniff_worker(self):
        """实际干活的子线程：抓包 15 秒"""
        window_id = time.strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(SAVE_DIR, f"window_{window_id}.pcap")

        print(f"[+] 启动新检测窗口: {window_id} (持续 {self.window_size}s)")

        # 使用 PcapWriter 比每次调用 wrpcap 性能高非常多
        pktdump = PcapWriter(file_name, append=True, sync=True)

        def packet_handler(pkt):
            pktdump.write(pkt)

        try:
            # stop_filter 是解决无法 Interrupt 关闭的核心！
            sniff(
                iface=TARGET_IFACE,
                filter=FILTER_STR,
                timeout=self.window_size,
                prn=packet_handler,
                store=0,  # 防止内存溢出
                stop_filter=lambda x: self.stop_event.is_set())
        finally:
            pktdump.close()

        # 窗口结束，如果文件不为空，则触发分析回调
        if os.path.exists(file_name) and os.path.getsize(
                file_name) > 24:  # 24字节是pcap文件头大小
            print(f"[-] 窗口 {window_id} 结束，准备分析。")
            if self.callback and not self.stop_event.is_set():
                self.callback(file_name)
        else:
            # 空文件清理
            if os.path.exists(file_name): os.remove(file_name)
