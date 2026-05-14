# 文件名：app.py
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import pandas as pd

from transfer import run_extraction
from predict_live_traffic import predict_traffic
from shark_manager import SlidingWindowSniffer

app = Flask(__name__)

UPLOAD_DIR = "uploads"

system_status = {
    "latest_file": "等待初始化...",
    "flow_count": 0,
    "table_html":
    "<div class='text-center text-muted py-5'>等待监控窗口完成抓取... (约需 15-30 秒)</div>",
    "is_running": False
}


def style_logic(row):
    status = str(row['Status'])

    if '[A]' in status or '[B]' in status:
        return ['background-color: #dc3545; color: white; font-weight: bold;'
                ] * len(row)

    if '[C]' in status:
        return [
            'background-color: #6f42c1; color: white; font-weight: bold;'
        ] * len(row)

    if '[E]' in status:
        return ['color: #856404; background-color: #fff3cd;'] * len(row)

    if '[D]' in status:
        return [
            'background-color: #e7f3ff; color: #004085; font-style: italic;'
        ] * len(row)

    return [''] * len(row)


def df_to_styled_html(summary_df):
    return summary_df.style.apply(style_logic, axis=1).to_html(
        classes='table table-hover table-bordered text-center align-middle',
        index=False)


def process_pcap(pcap_path):
    """处理 pcap 文件：提取特征 -> 模型预测 -> 返回 HTML 表格"""
    csv_path = pcap_path + ".csv"
    flow_count = run_extraction(pcap_path, csv_path)
    if flow_count == 0:
        return None, 0, "未提取到任何流记录"

    summary_df = predict_traffic(csv_path)
    if summary_df.empty:
        return None, 0, "分析失败：模型未产生结果"

    html_table = df_to_styled_html(summary_df)
    return html_table, flow_count, os.path.basename(pcap_path)


def on_window_finished(pcap_path):
    """实时监控窗口回调：处理抓包结果并更新全局状态"""
    try:
        html_table, flow_count, fname = process_pcap(pcap_path)
        if html_table is not None:
            system_status["table_html"] = html_table
        system_status["flow_count"] = flow_count
        system_status["latest_file"] = fname
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


@app.route('/api/upload', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "文件名为空"}), 400

    if not file.filename.lower().endswith(('.pcap', '.pcapng')):
        return jsonify({"status": "error", "message": "仅支持 .pcap / .pcapng 格式"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        html_table, flow_count, fname = process_pcap(save_path)
        return jsonify({
            "status": "success",
            "filename": fname,
            "flow_count": flow_count,
            "table_html": html_table
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"分析失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
