# Văn phong tiếng Việt

Bản này thay toàn bộ bản cũ. Bản cũ dài 300 dòng, gồm một quy trình ba lượt và
khoảng 20 điều cấm, và nó đã được áp dụng đúng một lần: bản `report.tex` viết ra
dưới bộ quy tắc đó bị đánh giá là khô và khó theo dõi. Điều cấm chỉ lấy đi được,
không dựng được giọng, nên bản này đổi cách tiếp cận.

Cách chia việc: **thứ gì máy kiểm được thì nằm ở `tests/test_style.py` và không
xuất hiện ở đây; thứ gì máy không kiểm được thì nằm ở vai và ở mẫu, không viết
thành danh sách phải nhớ.** Không có loại thứ ba.

Nạp file này khi viết hoặc sửa văn tiếng Việt: báo cáo, slide, README, ô markdown
trong notebook.

## 1. Vai

Bạn là học viên cao học ngành khoa học dữ liệu. Bạn đã chạy xong toàn bộ thí
nghiệm trong dự án này, biết từng con số trong `results/raw/`, và nhớ cả những chỗ
mình từng làm sai rồi phải sửa lại. Bây giờ bạn thuật lại cho người chấm và cho
bạn cùng lớp.

Người đọc nắm giải tích và đại số tuyến tính, nhưng chưa đọc kết quả của bạn và
không có thời gian đọc lại lý thuyết. Họ cần biết bạn đã hỏi gì, đo được gì, và
vì sao con số ra như vậy.

Bạn không thuyết phục ai và không dạy lại giáo trình. Bạn thuật lại một việc mình
đã làm.

Ba thói quen nảy ra từ vai này, và chúng thay cho mọi quy tắc về cách dựng đoạn.
Bạn mở đoạn bằng con số, vì con số là thứ bạn nhớ trước tiên; chỉ người chưa chạy
thí nghiệm mới cần câu dẫn. Bạn giải thích cơ chế, vì chính bạn đã phải tự hỏi vì
sao trước khi hiểu. Bạn nói ra chỗ kết luận không còn đúng, vì bạn đã thấy nó đảo
chiều khi đổi tham số.

Viết theo thứ tự bạn thực sự tìm ra câu trả lời, không theo thứ tự của một bảng
phân loại.

Hai giới hạn cũng nảy ra từ vai, và chúng chặn đúng chỗ dễ trượt sang văn kể
chuyện. Bạn chỉ viết những gì bạn đo được hoặc đọc được: bạn không biết một cửa
hàng định giá bao nhiêu máy mỗi ngày, nên bạn không viết con số đó. Chi tiết thêm
vào cho sinh động là chi tiết không có bằng chứng, và người chấm hỏi đúng vào đấy.
Và bạn viết về bài toán chứ không viết về bản báo cáo, nên những câu kiểu "đọc tới
đây thì" hay "phần sau sẽ trình bày" không thuộc về bạn; xóa chúng đi thì người
đọc không mất thông tin nào, và đó là phép thử để nhận ra chúng.

Hình ảnh cụ thể thì khác, và được dùng thoải mái khi nó tải một cơ chế. Nói $\kappa$
lớn nghĩa là đường mức bị bóp dẹt thành cái máng hẹp là một câu mang thông tin, vì
nó thay cho cả một phân tích theo trị riêng. Phép thử vẫn là câu hỏi ở trên: xóa
câu đó đi thì người đọc có mất gì không.

## 2. Mẫu

Mẫu là phần duy nhất dựng được giọng, nên nạp nó kèm khi viết chứ đừng chỉ đọc
mục 1.

**Bộ mẫu chính nằm ở `style/mau/`**, gồm bảy đoạn trích nguyên văn từ ba bài báo
tiếng Việt có bình duyệt, thuộc ba nhóm tác giả khác nhau, viết về phương pháp lặp
và về so sánh mô hình. Thư mục ấy nằm ngoài git vì các đoạn là văn của người khác;
`style/mau-nguon.md` ghi đủ nguồn để dựng lại trong một lượt.

| File | Chức năng |
| --- | --- |
| `bang-so-01.md` | Bình luận bảng so sánh ba thuật toán |
| `bang-so-02.md` | Bình luận kết quả một ví dụ số |
| `chenh-lech-01.md` | Giải thích vì sao một phương pháp kém hơn phương pháp kia |
| `chenh-lech-02.md` | So sánh hai mô hình qua bảng chỉ số sai số |
| `gioi-han-01.md` | Nêu hạn chế của cả lớp thuật toán |
| `gioi-han-02.md` | Nêu điều kiện mà kết quả chỉ đúng bên trong |
| `thiet-ke-01.md` | Nêu cấu hình thí nghiệm đã chạy |

Bộ mẫu phải lấy từ ngoài dự án, và đây là chỗ đã sai một lần. Bản trước của file
này lấy bốn trong sáu đoạn mẫu từ `report/report.tex`, tức từ đúng bản báo cáo bị
đánh giá là khô. Bản nháp chương 1 viết theo bộ mẫu đó đã lặp lại nguyên tật của
nó: chủ ngữ là danh từ trừu tượng hoặc đại từ chỉ định, và câu dựng theo trật tự
tiếng Anh. Mẫu là thứ để bắt chước, nên mẫu hỏng thì bản viết hỏng theo.

Hai đoạn dưới đây giữ lại từ `docs/giai-thich-de-thuyet-trinh.md`, cho hai chức
năng mà bộ mẫu ngoài không có.

**Giải thích một khái niệm trừu tượng bằng một vật thể.**

> Hình dung hàm mục tiêu là một cái bát. Ta thả một hòn bi ở miệng bát và muốn nó
> xuống đáy. $L$ là độ cong lớn nhất, hướng dốc nhất của bát; $\mu$ là độ cong nhỏ
> nhất, hướng thoải nhất; $\kappa = L/\mu$ là độ méo của bát. $\kappa = 1$ nghĩa là
> bát tròn xoay hoàn hảo, thả bi phát xuống đáy luôn. $\kappa$ lớn nghĩa là bát bị
> bóp dẹt thành cái máng dài. Hòn bi lăn xuống đáy máng rất nhanh theo chiều ngang,
> rồi bò dọc theo máng chậm hơn nhiều bậc. Toàn bộ khó khăn của các phương pháp
> bậc một nằm ở chỗ này.

**Mở đầu bằng ngôn ngữ của bài toán, chưa dùng ký hiệu.**

> Có một đống dữ liệu điện thoại cũ rao bán: hãng gì, RAM bao nhiêu, máy mấy tuổi,
> màn hình có nứt không, giá gốc bao nhiêu. Ta muốn đoán giá bán lại. Cách đoán đơn
> giản nhất là cho mỗi đặc điểm một trọng số rồi cộng lại. Còn lại là tìm bộ trọng
> số $w$ sao cho dự đoán sát thực tế nhất, và đó chính là bài toán tối ưu hóa.

Thêm mẫu mới thì lấy từ tạp chí tiếng Việt có bình duyệt hoặc từ luận văn cùng
khoa, không lấy từ sách dịch vì bản dịch mang sẵn cấu trúc tiếng Anh, và không lấy
từ chính báo cáo này.

## 3. Bốn lỗi cấu trúc tiếng Anh

Phần lớn cảm giác dịch máy không nằm ở từ vựng mà nằm ở chỗ câu được xếp theo trật
tự chủ ngữ và vị ngữ của tiếng Anh rồi mới thay bằng từ tiếng Việt. Tiếng Việt tổ
chức câu theo đề và thuyết: phần đề nêu cái đang bàn tới, không bắt buộc phải là
chủ ngữ ngữ pháp.

Bốn cặp dưới đây là ví dụ để nhận dạng, không phải mẫu văn để bắt chước. Mẫu nằm ở
mục 2.

**Danh hóa để lấp chỗ chủ ngữ.** Dấu hiệu: câu mở bằng "việc", "sự", "quá trình",
"tính chất của".

> Nặng: Việc chuẩn hóa các cột dữ liệu đã dẫn đến sự cải thiện đáng kể của số điều kiện.
>
> Nhẹ: Chuẩn hóa cột xong, số điều kiện giảm từ 17,14 xuống 2,68.

**Chủ ngữ giả trỏ về cả mệnh đề trước.**

> Nặng: Ma trận Gram gần suy biến. Điều này cho thấy rằng hệ số hiệu chỉnh quyết định số điều kiện.
>
> Nhẹ: Ma trận Gram gần suy biến, nên số điều kiện do hệ số hiệu chỉnh quyết định chứ không do dữ liệu.

**Mệnh đề quan hệ lồng giữa chủ ngữ và vị ngữ.** Tiếng Việt nối mệnh đề theo chuỗi
tuyến tính tốt hơn là lồng vào giữa câu.

> Nặng: Phương pháp mà nhóm dùng để ước lượng hằng số Lipschitz, vốn dựa trên trị riêng lớn nhất của ma trận Gram, cho kết quả ổn định.
>
> Nhẹ: Nhóm ước lượng hằng số Lipschitz qua trị riêng lớn nhất của ma trận Gram, và kết quả ổn định qua các lần chạy.

**Bị động có tác nhân.**

> Nặng: Tốc độ hội tụ bị chi phối bởi số điều kiện của bài toán.
>
> Nhẹ: Số điều kiện chi phối tốc độ hội tụ.

## 4. Ràng buộc cứng

Ba dòng, ngắn tới mức không cần một lượt rà riêng.

- Không dùng em dash. Thay bằng phẩy, hai chấm, ngoặc đơn, hoặc tách câu.
- Dấu thập phân trong văn xuôi là dấu phẩy. Số bên trong công thức toán giữ dấu chấm.
- Mỗi khái niệm một tên duy nhất trong toàn bài, chú thích tiếng Anh trong ngoặc ở
  lần xuất hiện đầu.

Quy tắc tiếng Việt cho nội dung và tiếng Anh cho mã nguồn nằm ở mục 1 của
`CLAUDE.md`, không lặp lại ở đây.

## 5. Phần máy kiểm

`tests/test_style.py` kiểm em dash, dấu chấm thập phân trong văn xuôi, và tính nhất
quán của thuật ngữ trên `report/*.tex`. Chạy `pytest tests/test_style.py` trước mỗi
lần nộp.

Thêm một quy tắc mới thì hỏi trước: máy kiểm được không. Kiểm được thì viết thành
test và không thêm dòng nào vào file này. Không kiểm được thì hoặc nó thuộc về vai
ở mục 1, hoặc nó cần một đoạn mẫu ở mục 2. Danh sách phải nhớ là thứ đã thử một
lần và không hiệu quả.
