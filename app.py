# 文件名：app.py
# version 514
import os
import threading
from flask import Flask, request, render_template, jsonify
import pandas as pd

from transfer import run_extraction
from predict_live_traffic import predict_traffic
from shark_manager import SlidingWindowSniffer

app = Flask(__name__)

system_status = {
    "latest_file": "等待初始化...",
    "flow_count": 0,
    "table_html":
    "<div class='text-center text-muted py-5'>等待监控窗口完成抓取... (约需 15-30 秒)</div>",
    "is_running": False
}


def on_window_finished(pcap_path):
    try:
        csv_path = pcap_path + ".csv"
        flow_count = run_extraction(pcap_path, csv_path)

        if flow_count > 0:
            summary_df = predict_traffic(csv_path)

            def style_logic(row):
                status = str(row['Status'])

                # 洪水/扫描 -> 红色
                if '[A]' in status or '[B]' in status:
                    return [
                        'background-color: #dc3545; color: white; font-weight: bold;'
                    ] * len(row)

                # 病毒横向探测 -> 紫色 (区分不同类型的攻击)
                if '[C]' in status:
                    return [
                        'background-color: #6f42c1; color: white; font-weight: bold;'
                    ] * len(row)

                # 可疑 -> 橙色
                if '[E]' in status:
                    return ['color: #856404; background-color: #fff3cd;'
                            ] * len(row)

                # 正常业务 -> 淡蓝色
                if '[D]' in status:
                    return [
                        'background-color: #e7f3ff; color: #004085; font-style: italic;'
                    ] * len(row)

                return [''] * len(row)

            # 生成 HTML (已修复原来的重复赋值bug)
            html_table = summary_df.style.apply(style_logic, axis=1).to_html(
                classes=
                'table table-hover table-bordered text-center align-middle',
                index=False)

            system_status["table_html"] = html_table
            system_status["flow_count"] = flow_count
            system_status["latest_file"] = os.path.basename(pcap_path)
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")


sniffer = SlidingWindowSniffer(callback=on_window_finished)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/toggle', methods=['POST'])
def toggle_sniffer():
    action = request.json.get('action')
    if action == 'start':
        sniffer.start()
        system_status["is_running"] = True
    elif action == 'stop':
        sniffer.stop()
        system_status["is_running"] = False

    return jsonify({
        "status": "success",
        "is_running": system_status["is_running"]
    })


@app.route('/api/latest_data', methods=['GET'])
def get_latest_data():
    return jsonify({
        "latest_file": system_status["latest_file"],
        "flow_count": system_status["flow_count"],
        "table_html": system_status["table_html"],
        "is_running": system_status["is_running"]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
