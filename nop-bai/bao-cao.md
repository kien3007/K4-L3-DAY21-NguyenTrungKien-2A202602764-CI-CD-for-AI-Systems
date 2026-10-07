# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Trung Kiên |
| MSSV | 2A202602764 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/kien3007/K4-L3-DAY21-NguyenTrungKien-2A202602764-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số `n_estimators=200`, `learning_rate=0.1`, `max_depth=5` đạt `f1_score` cao nhất (0.7149), vượt ngưỡng đảm bảo chất lượng 0.65 của hệ thống. Đáng chú ý, lần chạy 1 đạt accuracy cao nhất (0.8780 so với 0.8740 của lần 3), việc lần có accuracy cao nhất không trùng với lần có f1_score cao nhất minh chứng rằng accuracy bị chi phối mạnh bởi lớp đa số (thu nhập <= 50K chiếm hơn 75%), trong khi f1_score đo lường chuẩn xác sự cân bằng giữa precision và recall trên lớp thiểu số cần dự đoán. Ngoài ra, kết quả lần 2 (f1_score tụt xuống 0.6051) cho thấy đánh đổi rõ rệt: khi giảm learning_rate thì cần tăng n_estimators và độ sâu max_depth tương ứng để các cây bù trừ sai số hiệu quả.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng đáng kể: lớp thu nhập cao (>50K) chỉ chiếm 24,8%, trong khi lớp thu nhập thấp chiếm tới 75,2%. Hệ quả là một mô hình sơ sài luôn đoán nhãn "thu nhập thấp" cho mọi mẫu vẫn dễ dàng đạt accuracy 75,2%, tạo ra ảo tưởng về hiệu năng cao dù thực chất hoàn toàn vô dụng vì không bắt được bất kỳ trường hợp thu nhập cao nào. F1-score của lớp dương giải quyết triệt để vấn đề này nhờ tính trung bình điều hòa giữa Precision và Recall riêng trên lớp mục tiêu, phản ánh chính xác khả năng phát hiện đúng và đủ người có thu nhập cao. Ta tuyệt đối không dùng `average="macro"` hay `average="weighted"` vì các cách tính này sẽ để lớp đa số kéo điểm lên cao, che lấp sự yếu kém trên lớp thiểu số và làm vô hiệu hóa tiêu chuẩn của Quality Gate.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi cài đặt gói `pyyaml==6.0.10` khi cài dependencies | Phiên bản 6.0.10 không tồn tại trên PyPI do gõ nhầm phiên bản. | Sửa lại thành `pyyaml==6.0.1` trong file `requirements.txt`. |
| Lỗi `Permission denied (publickey)` khi SSH vào máy ảo EC2 | Máy ảo dùng AMI Amazon Linux 2023 nên username mặc định là `ec2-user` chứ không phải `ubuntu`. | Đổi lệnh kết nối sang `ssh -i <key.pem> ec2-user@<IP>`. |
| Lỗi unpickle mô hình `AttributeError: CyHalfBinomialLoss` trên EC2 | Phiên bản `scikit-learn` trên EC2 (1.6.1) không tương thích với bản lúc huấn luyện (1.4.2). | Cài đặt cố định chính xác `scikit-learn==1.4.2` trên máy ảo EC2. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới (nâng tổng số mẫu lên 44.722), f1_score cải thiện từ 0.7149 lên 0.7354 (+0.0205) và accuracy tăng từ 87.4% lên 88.2%. Việc tăng gấp đôi số lượng mẫu giúp Gradient Boosting học tốt hơn các biên quyết định phức tạp trên lớp thiểu số, mang lại hiệu năng cao hơn trên tập holdout. Quan trọng nhất, toàn bộ quá trình tải dữ liệu mới, chạy lại unit test, huấn luyện mô hình, kiểm tra quality gate và cập nhật server inference trên cloud VM đều diễn ra hoàn toàn tự động chỉ thông qua commit file DVC mà không cần bất kỳ can thiệp thủ công nào.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___

