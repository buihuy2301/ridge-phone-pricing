# Nguồn của bộ mẫu

`style/mau/` nằm ngoài git vì các đoạn trong đó trích từ bài báo của người khác.
File này ghi đủ thông tin để dựng lại bộ mẫu trong một lượt, nên mất thư mục kia
không phải mất bộ mẫu. Bản trước của quy trình không có file này, thư mục mẫu
biến mất khỏi máy, và phần sinh của bộ quy tắc mất theo.

| Tác giả | Bài | Nơi đăng | Năm | Tải tại |
| --- | --- | --- | --- | --- |
| Đỗ Đức Đông, Hoàng Xuân Huấn | Về biến thiên của vết mùi trong phương pháp ACO và các thuật toán mới | Tạp chí Tin học và Điều khiển học, T.27 S.3, tr. 263-274 | 2011 | `vjs.ac.vn/index.php/jcc/article/download/490/pdf(Vietnamese)/2411` |
| Phạm Quý Mười, Phan Thị Như Quỳnh | Một phương pháp chọn điểm khởi đầu trong giải thuật điểm trong cho bài toán quy hoạch tuyến tính | Tạp chí Khoa học và Công nghệ Đại học Đà Nẵng, Số 1(98), tr. 112-116 | 2016 | `jst-ud.vn/jst-ud/article/download/3553/3553/6120` |
| Trần Q. Cảnh, Vũ T. Phúc | Hàm Cobb-Douglas hay hàm Translog? Nghiên cứu thực nghiệm | HCMCOUJS Kinh tế và Quản trị Kinh doanh, 17(4), tr. 142-149 | 2021 | `journalofscience.ou.edu.vn/index.php/econ-vi/article/download/1886/1636` |

Bảy đoạn đã trích, mỗi đoạn một file trong `style/mau/`:

| File | Nguồn | Chức năng |
| --- | --- | --- |
| `bang-so-01.md` | Đỗ, Hoàng, tr. 273 | Bình luận bảng so sánh ba thuật toán |
| `bang-so-02.md` | Phạm, Phan, tr. 115 | Bình luận kết quả một ví dụ số |
| `chenh-lech-01.md` | Đỗ, Hoàng, tr. 272 | Giải thích vì sao một phương pháp kém hơn phương pháp kia |
| `chenh-lech-02.md` | Trần, Vũ, tr. 147 | So sánh hai mô hình qua bảng chỉ số sai số |
| `gioi-han-01.md` | Đỗ, Hoàng, tr. 271 | Nêu hạn chế của cả lớp thuật toán |
| `gioi-han-02.md` | Phạm, Phan, tr. 114 | Nêu điều kiện mà kết quả chỉ đúng bên trong |
| `thiet-ke-01.md` | Đỗ, Hoàng, tr. 273 | Nêu cấu hình thí nghiệm đã chạy |

Bảy đoạn ứng với ba nhóm tác giả, nên bộ mẫu không truyền tật riêng của một người
viết. Đoạn `chenh-lech-02.md` yếu hơn sáu đoạn còn lại vì nó mở bằng một cụm dẫn
trước khi vào số liệu; giữ nó vì nó là đoạn duy nhất viết đúng về RMSE, tức đúng
chủ đề của bài này.

Các đoạn được chép tay từ bản PDF nên có thể sai vài ký tự toán. Đối chiếu lại với
bản gốc trước khi trích dẫn ra ngoài phạm vi tham chiếu nội bộ.

Thiếu và nên bổ sung: chưa có đoạn nào lấy từ luận văn hay luận án, tức thể loại
sát bài này nhất. Thư viện `repository.vnu.edu.vn` và `ir.vnulib.edu.vn` có bộ sưu
tập toàn văn.
