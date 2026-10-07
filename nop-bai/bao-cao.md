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

**Lý do:** Bộ tham số này đạt `f1_score` cao nhất (0.7149), vượt qua ngưỡng Quality Gate 0.65. Lần 1 có accuracy cao hơn (0.8780 so với 0.8740), chứng minh accuracy bị chi phối bởi lớp đa số (thu nhập <= 50K chiếm 75.2%), trong khi F1 phản ánh chính xác hiệu năng trên lớp thiểu số cần phát hiện. Việc giảm learning_rate ở lần 2 đòi hỏi tăng số lượng cây và độ sâu tương ứng để bù trừ sai số.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult mất cân bằng khi lớp thu nhập cao (>50K) chỉ chiếm 24.8%. Mô hình đoán mò toàn bộ nhãn thu nhập thấp vẫn đạt accuracy 75.2%, tạo ảo giác chất lượng nhưng hoàn toàn vô dụng. F1-score giải quyết triệt để vấn đề này nhờ tính trung bình điều hòa giữa Precision và Recall riêng trên lớp mục tiêu. Không dùng macro hay weighted average vì lớp đa số sẽ kéo điểm trung bình lên cao, che giấu sự kém hiệu quả trên nhóm khách hàng tiềm năng.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi cài đặt gói `pyyaml==6.0.10` | Phiên bản 6.0.10 không tồn tại trên PyPI. | Cập nhật lại thành `pyyaml==6.0.1` trong `requirements.txt`. |
| Lỗi `Permission denied (publickey)` khi SSH | Máy ảo dùng AMI Amazon Linux 2023 có username mặc định là `ec2-user`. | Đổi lệnh kết nối sang `ssh -i <key.pem> ec2-user@<IP>`. |
| Lỗi unpickle mô hình `AttributeError` trên EC2 | Bản `scikit-learn` trên EC2 (1.6.1) lệch so với bản huấn luyện (1.4.2). | Cài đặt cố định `scikit-learn==1.4.2` trên môi trường EC2. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Bổ sung 22.361 mẫu dữ liệu mới giúp mô hình học tốt hơn các biên phân tách phức tạp, cải thiện F1 từ 0.7149 lên 0.7354 (+0.0205) và accuracy từ 87.4% lên 88.2%. Quan trọng nhất, chu trình đồng bộ dữ liệu DVC, chạy unit test, huấn luyện, kiểm tra chất lượng và cập nhật dịch vụ suy luận trên EC2 đều được kích hoạt tự động qua Git commit mà không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Kết nối và đồng bộ tự động toàn bộ tham số, metrics và mô hình lên máy chủ MLflow hosted trên DagsHub trong quy trình GitHub Actions qua biến môi trường xác thực.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng từ 0.1 đến 0.9, xác định ngưỡng tối ưu là 0.30 giúp F1 tăng từ 0.7354 lên 0.7537 so với ngưỡng mặc định 0.50.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Xuất ma trận nhầm lẫn và chỉ số phân lớp ra `outputs/detail.txt` thành artifact. Đối với bài toán tiếp thị dịch vụ cao cấp, sai lầm bỏ sót khách hàng thu nhập cao (recall thấp) tốn kém hơn nhiều so với việc tiếp cận nhầm người thu nhập thấp (precision thấp) vì giá trị vòng đời khách hàng vượt trội chi phí tiếp thị.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: Đọc F1 của mô hình trước đó trên S3; nếu mô hình mới bị suy giảm hiệu năng (F1 mới < F1 cũ), pipeline sẽ dừng lại ở Quality Gate và hủy release để bảo đảm an toàn.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Kiểm tra phân phối tập huấn luyện đạt 24.78% nhãn dương (sát mốc chuẩn 24.80%) và tự động in cảnh báo nếu độ lệch vượt quá 5%.
