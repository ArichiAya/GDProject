import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pandas as pd

# 导入你修改好的两个脚本的函数
from transfer import run_extraction
from predict_live_traffic import predict_traffic

app = Flask(__name__)

# 设置上传目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. 获取上传的文件
        if 'pcap_file' not in request.files:
            return "未找到文件部分"
        file = request.files['pcap_file']

        if file.filename == '':
            return "未选择文件"

        if file:
            filename = secure_filename(file.filename)
            pcap_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            csv_path = pcap_path + ".csv"

            # 保存上传的 Pcap
            file.save(pcap_path)

            try:
                # 2. 调用 transfer.py 提取特征
                flow_count = run_extraction(pcap_path, csv_path)

                # 3. 调用 predict_live_traffic.py 进行预测
                result_df = predict_traffic(csv_path)

                # 4. 将 DataFrame 转换为 HTML 表格用于网页展示
                # 将恶意流量行加上高亮 class
                table_html = result_df.to_html(
                    classes='table table-striped table-hover',
                    index=False,
                    justify='center')

                return render_template('index.html',
                                       table=table_html,
                                       filename=filename,
                                       flow_count=flow_count)
            except Exception as e:
                return f"处理过程中发生错误: {str(e)}"

    # GET 请求：直接显示主页
    return render_template('index.html', table=None)


if __name__ == '__main__':
    # 启动 Web 服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
