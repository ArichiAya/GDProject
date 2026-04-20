# import csv

# with open(
#         "./src/Malicious-CSVs/Malicious-CSVs/Network-CSVs/FinalVersion-PerSample/Backdoor/11_Backdoor_0a57e866ca2f4e43f8d2c3b1e4fa84302673046dc637de823fad5ed1c2e1ee69.csv",
#         "r") as csvfile:
#     csv_reader = csv.reader(csvfile)
#     row_text = next(csv_reader)
#     row_text = next(csv_reader)
#     print(row_text)
from scapy.all import get_if_list, show_interfaces

# 打印所有网卡的详细对照表（包括名称、IP、MAC）
show_interfaces()
