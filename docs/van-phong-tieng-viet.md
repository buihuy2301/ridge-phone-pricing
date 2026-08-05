# Văn phong tiếng Việt

File này là bản đầy đủ của quy tắc văn phong tóm tắt ở mục 2 của `CLAUDE.md`. Đọc
trước khi viết hoặc sửa bất kỳ đoạn văn tiếng Việt nào: báo cáo, slide, README, ô
markdown trong notebook.

Dùng theo hai lượt. Lượt viết bám mục 1 tới mục 5, tức bám vào mẫu và công thức.
Lượt rà soát dùng mục 7, tức danh sách các lỗi đã gặp. Không dùng mục 7 làm tài
liệu hướng dẫn viết: danh sách cấm chỉ thu hẹp vùng được phép, nó không chỉ ra
văn đích trông như thế nào, và viết theo nó sẽ ra một chuỗi câu ngắn rời rạc.

## 1. Văn đích

Báo cáo này hướng tới văn giáo trình toán ứng dụng: người viết đã làm thí nghiệm,
đang thuật lại cho một người đọc có nền tảng và muốn biết con số nào dẫn tới kết
luận nào. Sáu đặc điểm của thứ văn đó:

- Đoạn mở bằng dữ kiện, thường là một con số kèm mốc so sánh, chứ không mở bằng
  câu giới thiệu về nội dung sắp trình bày.
- Mỗi đoạn có ít nhất một câu ghép, tức câu có mệnh đề phụ chỉ nguyên nhân, điều
  kiện, thời điểm hoặc tương phản. Câu ghép này là chỗ đặt lập luận; các câu đơn
  quanh nó đặt dữ kiện.
- Câu nối với câu bằng quan hệ logic hiện rõ trên bề mặt: "vì", "do đó", "nhờ
  vậy", "khi", "trong khi", "ngược lại", "đổi lại", "miễn là".
- Con số luôn đi cùng mốc để so sánh, dạng "0,076 giây, ít hơn 10 lần so với
  gradient descent", không đứng trơ một mình.
- Việc mình làm thì xưng "nhóm"; tính chất của bài toán thì để bài toán làm
  chủ ngữ.
- Giới hạn của kết luận nêu ngay tại chỗ, thường bằng một mệnh đề điều kiện đóng
  đoạn: "nhận xét đó sẽ đảo chiều nếu $d$ lớn tới mức không tính được trị riêng
  lớn nhất".

## 2. Bộ mẫu

Bốn đoạn dưới đây lấy nguyên từ `report/report.tex`, là các đoạn đọc tự nhiên
nhất trong bản hiện tại. Khi viết đoạn mới, đối chiếu nhịp và cách nối câu với
các mẫu này trước, rồi mới rà theo mục 7.

**Mẫu 1, giải thích một quyết định về thiết kế thí nghiệm.**

> Toàn bộ thí nghiệm chạy trên một mẫu ngẫu nhiên 200 000 bản ghi, chia thành
> 160 000 điểm huấn luyện và 40 000 điểm kiểm tra. Việc lấy mẫu là quyết định về
> chi phí tính toán chứ không phải về thống kê: với toàn bộ một triệu bản ghi,
> một lần tính gradient mất 271 ms thay vì 39 ms, và toàn bộ lưới tham số sẽ mất
> hàng chục giờ.

Câu đầu 24 tiếng đặt dữ kiện, câu sau 46 tiếng mang toàn bộ lập luận. Đúng một
dấu hai chấm, dẫn vào phần định lượng cho mệnh đề đứng trước nó. Đoạn này dài hơn
hạn 40 từ ở mục 7 nhưng vẫn dễ đọc, vì mệnh đề sau dấu hai chấm chỉ mở rộng ý đã
nêu chứ không thêm ý mới.

**Mẫu 2, so sánh hai phương án bằng cấu trúc song song.**

> Nếu lấy tích trực tiếp trên thang gốc, tức nhân `original_price` (cỡ $10^{5}$)
> với `screen_size_inches` (cỡ 6), thì ma trận thiết kế thu được có $L = 17{,}14$
> và 48 trị riêng dưới $10^{-6}$. Nếu lấy tích trên các cột đã chuẩn hóa thì
> $L = 2{,}68$ và số trị riêng dưới $10^{-6}$ trở về đúng 7, bằng mức của khối
> cột gốc. Không gian cột sinh ra vẫn như cũ, chỉ cách biểu diễn thay đổi, và số
> điều kiện giảm theo, 6,4 lần.

Hai câu "Nếu... thì..." song song nhau, cùng độ dài, cùng thứ tự đại lượng, nên
người đọc so sánh được hai cột số mà không cần bảng. Câu thứ ba ngắn hơn hẳn và
chốt lại cơ chế.

**Mẫu 3, chuỗi suy luận dẫn tới một đẳng thức.**

> Ma trận Gram $\tfrac{1}{n} X^{\top} X$ có hạng 273 trên 280, và ngay cả khi bỏ
> qua 7 hướng suy biến đó, trị riêng kế tiếp cũng chỉ cỡ $2 \cdot 10^{-8}$. Nói
> cách khác $\lambda_{\min}(\tfrac{1}{n} X^{\top} X) \approx 0$, nên
> $\mu = \lambda$ và $\kappa = L / \lambda$. Hệ số hiệu chỉnh là đại lượng duy
> nhất quyết định số điều kiện của bài toán, chứ không phải cấu trúc của dữ liệu.
> Chương 8 khai thác tính chất này.

Ba bậc rõ rệt: số đo, hệ quả toán học, phát biểu tổng quát. Câu cuối bảy tiếng
làm nhiệm vụ chỉ đường, mang thông tin về vị trí nên hợp lệ.

**Mẫu 4, giải thích một chênh lệch bất ngờ giữa hai phương pháp.**

> Trên trục thời gian ở hình 12, Newton-CG chậm hơn Newton 25 lần, vì với
> $d = 280$, chi phí $O(d^{3}) \approx 2 \cdot 10^{7}$ của phân rã Cholesky nhỏ
> hơn nhiều so với chi phí một loạt tích Hessian nhân vector, mỗi tích tốn
> $O(nd) \approx 4{,}5 \cdot 10^{7}$. Newton-CG chỉ có lợi khi $d$ lớn tới mức
> không lập được ma trận $d \times d$.

Một câu dài mang cả hiện tượng lẫn nguyên nhân, nối bằng "vì" chứ không tách
thành hai câu rồi thêm "Nguyên nhân là". Câu sau nêu điều kiện mà kết luận đảo
chiều.

## 3. Trước khi viết

Bảng số và hình vẽ không tự sinh ra đoạn văn. Trước câu đầu tiên, xác định ý mà
đoạn cần chứng minh rồi mới chọn số liệu phục vụ ý đó; làm ngược lại, tức đọc
bảng trước rồi thuật lại những gì nhìn thấy, sẽ ra một đoạn liệt kê không có
trọng tâm.

Mỗi đoạn trả lời đúng một câu hỏi, và trước khi viết cần trả lời được bốn câu sau
về nó:

- Con số nào trong bảng hoặc hình đáng chú ý nhất?
- Con số đó dẫn tới kết luận gì?
- Cơ chế nào giải thích kết luận đó?
- Kết luận còn đúng trong điều kiện nào?

Chưa trả lời được cả bốn thì chưa đủ nguyên liệu để viết, và việc cần làm là đọc
lại kết quả hoặc chạy thêm thí nghiệm chứ không phải viết một đoạn mô tả bảng để
lấp chỗ.

Điểm xuất phát là kết luận cần rút ra, không phải hàng đầu tiên của bảng. Khi đã
có kết luận, giữ lại đúng những số liệu đủ để người đọc kiểm chứng nó, còn các số
liệu khác thì dành cho đoạn khác hoặc để nguyên trong bảng, vì bảng đã in đầy đủ
và người đọc tra được. Một đoạn tốt thường bỏ qua nhiều số hơn số nó dùng.

### 3.1. Mỗi đoạn chỉ chứng minh một ý

Một đoạn gộp nhiều hiện tượng thì không chứng minh được hiện tượng nào.

> Sai: Bảng 5 cho thấy Newton nhanh hơn gradient descent, Newton-CG chậm hơn
> Newton, backtracking làm tăng số lần đánh giá hàm, và chuẩn hóa cải thiện số
> điều kiện.

Bốn mệnh đề trên cần bốn cơ chế khác nhau, nên bản đúng là bốn đoạn cùng đọc từ
bảng 5 nhưng mỗi đoạn chỉ giữ các cột phục vụ ý của nó: một đoạn giải thích
Newton, một đoạn giải thích Newton-CG, một đoạn giải thích backtracking, một đoạn
giải thích chuẩn hóa.

Mốc để quyết định tách hay không là số cơ chế chứ không phải số kết luận. Hai kết
luận dựa trên cùng một cơ chế thì viết chung được, còn hai kết luận cần hai cơ
chế khác nhau thì nằm ở hai đoạn, ngay cả khi cùng đọc từ một bảng.

### 3.2. Viết theo chuỗi suy luận, không theo thứ tự đọc bảng

Người đọc không cần biết nhóm đã nhìn bảng theo thứ tự nào, họ chỉ cần biết
vì sao kết luận đúng. Thứ tự trong đoạn vì thế là thứ tự của lập luận: dữ kiện đủ
để chứng minh, cơ chế nối dữ kiện với kết luận, rồi kết luận hoặc giới hạn của
nó. Mục 4 khai triển ba phần này thành công thức viết đoạn.

Thứ tự cần tránh là thứ tự đọc bảng, tức đọc số thứ nhất, số thứ hai, số thứ ba,
cuối cùng mới tới kết luận. Đoạn viết theo thứ tự đó bắt người đọc giữ ba con số
trong đầu khi còn chưa biết chúng dùng để làm gì, và dồn toàn bộ phần suy luận
vào một câu cuối. Đoạn văn phải đọc như một lời giải thích, không phải lời thuyết
minh bảng.

## 4. Công thức cho đoạn phân tích kết quả

Phần lớn báo cáo là các đoạn bình luận một bảng hoặc một hình. Một đoạn như vậy
gồm ba phần, theo thứ tự:

1. **Dữ kiện.** Con số đọc từ bảng hoặc hình, kèm nguồn bằng `\ref` và kèm mốc so
   sánh. Đây là câu đầu tiên của đoạn, không có câu dẫn nào đứng trước.
2. **Cơ chế.** Vì sao con số ra như vậy. Phần này viết thành câu ghép, nối vào
   dữ kiện bằng "vì", "do", "khi". Nếu cơ chế cần tới một công thức thì dẫn công
   thức bằng `\eqref` ngay trong câu.
3. **Hệ quả hoặc giới hạn.** Kết luận rút ra, hoặc điều kiện mà kết luận không
   còn đúng. Một câu, đặt cuối đoạn.

Ba phần không cần ba câu. Mẫu 4 gộp dữ kiện và cơ chế vào một câu, mẫu 2 dùng hai
câu song song cho phần dữ kiện. Phần được phép lược là phần cơ chế, khi nó đã
trình bày ở mục trước và chỉ cần nhắc lại bằng một cụm.

Khi một bảng cho nhiều kết luận độc lập, mỗi kết luận là một đoạn riêng có đủ ba
phần, thay vì gộp thành một đoạn dài. Câu dẫn kiểu "Bảng 3 cho ba kết luận" chỉ
dùng một lần cho cả cụm đoạn, và chỉ khi thật sự có ba đoạn theo sau.

## 5. Nhịp câu và liên kết

Lỗi nặng nhất của các bản nháp gần đây không còn là câu rườm mà là câu đều: một
chuỗi câu trần thuật ngắn, mỗi câu một mệnh đề chính, không câu nào nối vào câu
nào. Đoạn dưới đây, lấy từ `report.tex`, có năm câu, câu dài nhất 20 tiếng, và
chỉ một câu có mệnh đề phụ:

> Đại lượng cần báo cáo ở đây là số lần đánh giá hàm mục tiêu mà line search tiêu
> tốn trên mỗi vòng lặp. Phép đo phải thực hiện trong giai đoạn thuật toán còn
> tiến triển. Khi $f$ đã chạm giới hạn số học, không độ dài bước nào còn thỏa mãn
> điều kiện Armijo. Line search vì thế tiêu hết ngân sách ở mọi vòng lặp, làm con
> số đo được tăng lên gấp năm lần. Bảng 4 lấy trong 100 vòng lặp đầu.

Bản sửa, gộp theo quan hệ logic và bỏ câu dẫn ở đầu:

> Bảng 4 đo số lần đánh giá hàm mục tiêu mà line search tiêu tốn trên mỗi vòng
> lặp, lấy trong 100 vòng lặp đầu. Phép đo chỉ có nghĩa trong giai đoạn thuật
> toán còn tiến triển, vì khi $f$ đã chạm giới hạn số học thì không độ dài bước
> nào còn thỏa mãn điều kiện Armijo, line search tiêu hết ngân sách ở mọi vòng
> lặp và con số đo được tăng lên gấp năm lần.

Ba mốc cần giữ khi viết:

- Không quá ba câu đơn liên tiếp. Câu thứ tư phải có mệnh đề phụ, hoặc phải nối
  vào câu trước bằng một liên từ chỉ quan hệ.
- Mỗi đoạn có ít nhất một câu từ 25 tiếng trở lên. Đoạn toàn câu dưới 20 tiếng
  đọc như bảng liệt kê.
- Hai câu liên tiếp không mở đầu bằng cùng một khuôn. Kiểm tra riêng các khuôn
  hay lặp: chủ ngữ là tên bảng hoặc hình, "Với...", "Khi...", "Số liệu...".

Liên từ nên dùng, phân theo quan hệ:

| Quan hệ | Từ nối |
| --- | --- |
| Nguyên nhân | vì, do, bởi |
| Hệ quả | nên, do đó, nhờ vậy, vì thế |
| Điều kiện | nếu, khi, miễn là, trừ khi |
| Tương phản | nhưng, trong khi, ngược lại, đổi lại, tuy nhiên |
| Bổ sung | ngoài ra, hơn nữa, đồng thời |
| Diễn giải | tức là, nói cách khác |

Quy tắc tách câu dài ở mục 7 nhắm vào câu chứa ba ý độc lập nối bằng dấu phẩy,
không nhắm vào câu ghép có mệnh đề phụ. Mốc để quyết định là số ý, không phải số
tiếng: một câu 45 tiếng gồm một ý chính và hai mệnh đề bổ nghĩa cho nó thì giữ
nguyên, còn một câu 30 tiếng gồm ba ý ngang hàng thì tách.

## 6. Trước và sau

Ba cặp dưới đây là các lỗi lặp lại nhiều lần nhất trong bản nháp.

**Câu dẫn thay cho dữ kiện.**

> Sai: Backtracking cho phương pháp tăng tốc cần điều kiện chặt hơn. Đây là kết
> quả đáng chú ý nhất của mục này. Với $\alpha = 0{,}3$ và $t_0 = 1$, phương pháp
> không hội tụ, dừng ở sai số $5{,}5 \cdot 10^{-5}$.
>
> Đúng: Với $\alpha = 0{,}3$ và $t_0 = 1$, phương pháp tăng tốc không hội tụ mà
> dừng ở sai số $5{,}5 \cdot 10^{-5}$. Điều kiện Armijo thông thường không đủ cho
> phương pháp này.

**Đại từ chỉ định mở đầu câu.**

> Sai: Lô kích thước 1 phân kỳ thành `NaN` ngay trong lượt duyệt đầu tiên. Đây là
> hệ quả trực tiếp của tỉ lệ 1159 lần nói trên.
>
> Đúng: Lô kích thước 1 phân kỳ thành `NaN` ngay trong lượt duyệt đầu tiên, đúng
> như tỉ lệ 1159 lần dự báo.

**Tiêu đề `\paragraph` là mệnh đề.**

> Sai: `\paragraph{Newton vượt trội hoàn toàn trên bài toán này, và đó là điều
> hiển nhiên.}`
>
> Đúng: `\paragraph{Phương pháp Newton.}` rồi mở đầu đoạn bằng dữ kiện: "Toàn bộ
> lời giải mất 0,076 giây, ít hơn 10 lần so với thời gian gradient descent cần để
> đạt $10^{-6}$."

## 7. Danh sách rà soát

Dùng sau khi đã viết xong đoạn, không dùng khi đang viết.

**Giàn giáo và câu thừa**

- Bỏ câu chỉ bình luận rằng nội dung sắp viết là quan trọng: "Điểm cần làm rõ
  trước là", "Một tính chất cần nêu rõ ngay ở đây", "Con số đáng so sánh là",
  "Dưới đây là bốn phép thử quan trọng nhất". Phép thử: xóa câu đó đi rồi xem
  người đọc có mất thông tin nào không.
- Giữ câu chỉ đường mang thông tin về vị trí: "Mục 8 khai thác tính chất này",
  "Cách quy đổi hàm mục tiêu của thư viện nằm ở mục 7".
- Bỏ câu kết đoạn dạng cách ngôn: "Tham số mặc định của thư viện không phải lúc
  nào cũng phù hợp với bài toán cụ thể". Nếu cần câu tổng kết thì gắn nó với con
  số cụ thể.
- Bỏ khung "Lý do là", "Nguyên nhân là". Nối thẳng bằng "vì", "do", "nên". Chỉ
  giữ khung khi phần giải thích dài tới mức phải tách thành câu riêng.

**Câu và chủ ngữ**

- Không mở đầu câu bằng "Đây là", "Điều này", "Điều đó" trỏ về cả mệnh đề đứng
  trước. Gộp vào câu trước, hoặc nêu đích danh danh từ được trỏ.
- Không viết câu cụt thiếu vị ngữ: "Ba kết luận.", "Hai cách xử lý, cả hai đều
  hội tụ.". Viết đủ: "Từ bảng 3 có thể rút ra ba kết luận."
- Chủ ngữ của việc cài đặt, chạy thí nghiệm, chọn tham số là "nhóm" hoặc
  lược bỏ, không phải "báo cáo" và không phải "chúng tôi". Khi câu nói về nội dung của chính văn bản thì
  lấy văn bản làm chủ ngữ lại đúng: "Báo cáo gồm tám mục", "Bảng 3 cho thấy ba
  kết luận".
- Tách câu chứa từ ba ý độc lập trở lên nối bằng dấu phẩy. Xem mốc ở mục 5.
- Hạn chế bị động "được" khi đã có chủ thể rõ ràng: "Toàn bộ thí nghiệm được thực
  hiện trên một mẫu ngẫu nhiên" viết thành "Toàn bộ thí nghiệm chạy trên một mẫu
  ngẫu nhiên". Giữ "được" khi nó mang nghĩa khả năng: "tính được", "đo được",
  "không phân biệt được".
- Không dùng bị động có tác nhân "bị chi phối bởi X". Đảo thành "X chi phối".
- Không để giới từ treo cuối câu: "trường hợp xấu nhất mà cận được thiết kế cho"
  viết thành "trường hợp xấu nhất mà cận nhắm tới".
- Tiêu đề mục và tiểu mục dùng lối danh hóa hoặc động từ chủ động: "Tạo biến
  tương tác trên cột đã chuẩn hóa", không viết "Biến tương tác phải được tạo
  trên cột đã chuẩn hóa".

**Khuôn lặp**

- Dấu hai chấm chỉ dùng khi dẫn vào liệt kê, định nghĩa, hoặc phần định lượng cho
  mệnh đề đứng trước. Trong bản nháp gần nhất, 35 trên khoảng 242 câu kết thúc
  bằng dấu hai chấm, tức cứ bảy câu lại có một câu theo nhịp "khẳng định, hai
  chấm, giải thích".
- Công thức tương phản "không phải X mà là Y", "không chỉ X mà còn Y", "X chứ
  không phải Y" dùng nhiều nhất một lần mỗi mục.
- Khung đếm "Bảng N cho ba kết luận" dùng nhiều nhất một lần mỗi chương.

**Từ vựng**

- Không dùng em dash. Thay bằng dấu phẩy, dấu hai chấm, dấu ngoặc đơn, hoặc tách
  thành câu riêng.
- Không dùng từ cường điệu: "cực kỳ", "siêu", "tuyệt vời", "bí quyết", "chìa
  khóa", "đột phá", "must-know".
- Không dùng câu cảm thán và emoji trong phần nội dung. Emoji chỉ dùng làm ký
  hiệu trong bảng trạng thái nếu thực sự cần.
- Không trộn văn nói: "cho có", "thắng rõ", "thắng tuyệt đối", "đụng tới", "khó
  bị đánh bại", "đổi chác", "hội tụ thật", "chỉ đơn giản là", "tỏ ra vượt trội".
- Không viết theo lối bán hàng hay thuyết phục: "nắm được điều này là nắm được
  nửa môn học", "chỉ cần 5 phút là hiểu".
- Không chen từ tiếng Anh khi đã có từ tương đương: viết "trùng nhau từng bit",
  không viết "trùng nhau bit-by-bit". Thuật ngữ giữ nguyên tiếng Anh phải dùng
  nhất quán cả bài, chẳng hạn `line search`, `backtracking`, `epoch`.
- Không dùng hệ từ theo lối tiếng Anh: "cái giá phải trả là nhỏ" viết thành "cái
  giá phải trả không đáng kể".
- Kiểm tra các cụm dễ đọc nhầm do hai từ chức năng đứng cạnh nhau: "hệ số quan
  sát được ước lượng bằng cách", "ba quy tắc giảm dần đều đưa sai số".

**Thuật ngữ, số, bằng chứng**

- Mỗi khái niệm một tên duy nhất trong toàn bài. Chọn "lượt duyệt dữ liệu" thì
  chú thích "(epoch)" ở lần xuất hiện đầu tiên rồi dùng thống nhất.
- Dấu thập phân trong văn xuôi là dấu phẩy. `preamble.tex` khai báo
  `\sisetup{output-decimal-marker={,}}` để `siunitx` in ra cùng dạng. Số trong
  công thức toán giữ dấu chấm theo quy ước quốc tế.
- Mọi kết luận rút ra từ thí nghiệm phải kèm số liệu hoặc biểu đồ tương ứng. Chưa
  có bằng chứng thì ghi rõ đó là dự đoán.

## 8. Ranh giới với mã nguồn

Quy tắc ngôn ngữ ở mục 1 của `CLAUDE.md` cấm tiếng Việt trong mã nguồn, kể cả
comment và docstring.

Ví dụ đúng:

```python
def backtracking_line_search(problem, w, grad, alpha=0.3, beta=0.8, t0=1.0):
    """Armijo backtracking. Returns (step_size, n_function_evals)."""
    t = t0
    n_evals = 0
    # Shrink the step until the sufficient decrease condition holds.
    while problem.f(w - t * grad) > problem.f(w) - alpha * t * grad @ grad:
        t *= beta
        n_evals += 1
    return t, n_evals
```

Ví dụ sai:

```python
def tim_buoc_nhay(bai_toan, w, grad):
    # thu nho buoc cho den khi thoa dieu kien giam du
    ...
```
