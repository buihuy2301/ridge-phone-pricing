# Giải thích dân dã toàn bộ bài tập, để học và thuyết trình

File này không phải báo cáo mà là bản nói cho dễ hiểu, dùng để tự học trước khi
đứng lớp. Mọi con số trong đây lấy từ `report/report.tex` và
`results/raw/summary_all_methods.csv`, nên nắm được ý ở đây thì trả lời được câu
hỏi về báo cáo.

Cách dùng: đọc mục 1 tới 3 để nắm câu chuyện, mục 4 tới 9 để nắm từng thuật toán,
mục 10 để luyện trả lời phản biện, mục 11 là kịch bản nói 20 phút.

---

## 1. Bài này thực ra đang làm gì

Có một đống dữ liệu điện thoại cũ rao bán: hãng gì, RAM bao nhiêu, máy mấy tuổi,
màn hình có nứt không, giá gốc bao nhiêu. Ta muốn đoán giá bán lại.

Cách đoán đơn giản nhất là cho mỗi đặc điểm một trọng số rồi cộng lại:

$$
\text{giá dự đoán} = w_1 \cdot \text{RAM} + w_2 \cdot \text{tuổi máy} + \dots
$$

Còn lại là tìm bộ trọng số $w$ sao cho dự đoán sát thực tế nhất, và đó chính là
bài toán tối ưu hóa.

Slide đầu cần nói rõ một điều: môn này không chấm việc dự đoán giá giỏi hay dở mà
chấm cách đi tìm $w$. Bài toán định giá chỉ là chỗ để có một hàm mục tiêu thật,
đủ lớn, rồi đem bảy thuật toán ra chạy đua trên đó.

Vì vậy có một nguyên tắc xuyên suốt: nhóm đóng băng hàm mục tiêu trước khi chạy
thí nghiệm. Dữ liệu $X$, $y$ và hệ số $\lambda$ chốt xong là không ai sửa nữa, vì
nếu giữa chừng có người sửa dữ liệu thì mọi biểu đồ trước đó mất giá trị so
sánh, do các đường không còn chung một đích.

### Quy mô

| Thứ | Số |
| --- | --- |
| Dữ liệu gốc trên Kaggle | 1.000.000 dòng, 28 cột |
| Mẫu dùng để chạy | 200.000 dòng |
| Tập huấn luyện $n$ | 160.000 |
| Tập kiểm tra | 40.000 |
| Số thuộc tính $d$ sau mã hóa | 280 |

Nhóm lấy mẫu 200.000 dòng thay vì chạy cả triệu, vì trên toàn bộ dữ liệu một lần
tính gradient mất 271 ms thay vì 39 ms. Chênh lệch bảy lần ấy tự nó nhỏ, nhưng cả
lưới tham số có hàng chục nghìn vòng lặp nên nó nhân lên thành hàng chục giờ.
Động cơ ở đây là chi phí tính toán chứ không phải cỡ mẫu thống kê, và nếu hội
đồng hỏi thì phải trả lời đúng như vậy.

---

## 2. Hàm mục tiêu, giải thích bằng lời

$$
f(w) = \underbrace{\frac{1}{2n} \| Xw - y \|_2^2}_{\text{sai bao nhiêu}} + \underbrace{\frac{\lambda}{2} \| w \|_2^2}_{\text{phạt vì tham số to}}
$$

Vế đầu là trung bình bình phương sai số: dự đoán lệch càng nhiều thì bị cộng càng
nặng. Vế sau là **hiệu chỉnh Ridge**, phạt mô hình nếu trọng số quá lớn. Không có
vế phạt thì mô hình dễ bám vào nhiễu của tập huấn luyện.

Vài chi tiết nhỏ nhưng hay bị hỏi:

- **Vì sao chia cho $2n$?** Chia cho $n$ để hằng số Lipschitz $L$ không phụ thuộc
  vào số mẫu, nhờ vậy so sánh với SGD mới công bằng. Chia thêm cho 2 để khi lấy
  đạo hàm thì số 2 triệt tiêu, công thức gradient sạch.
- **Hệ số chặn $b$ đâu?** Đã khử. Nhóm chuẩn hóa mọi cột của $X$ về trung bình 0
  và trừ trung bình khỏi $y$, khi đó nghiệm tối ưu có $b^* = 0$ nên bỏ hẳn biến
  này đi được. Nếu giữ $b$ như một cột hằng thì nó cũng bị phạt $\lambda$, mà
  phạt hệ số chặn là sai quy ước vì nghiệm sẽ đổi khi ta dịch $y$ lên xuống.
- **Vì sao lấy logarit của giá?** Giá lệch phải rất mạnh: trung vị 18.555, trung
  bình 22.598, lớn nhất 160.257. Giữ thang gốc thì vài trăm máy đắt tiền chiếm
  hết hàm mục tiêu. Hệ quả cần nhớ: mọi RMSE trong báo cáo đọc trên thang
  logarit, không quy ra tiền được.

### Gradient và Hessian

$$
\nabla f(w) = \frac{1}{n} X^\top (Xw - y) + \lambda w,
\qquad
\nabla^2 f(w) = \frac{1}{n} X^\top X + \lambda I
$$

Gradient là hướng dốc lên. Muốn đi xuống thì đi ngược lại nó. Hessian là độ cong,
nó cho biết mặt phẳng cong nhanh cỡ nào theo từng hướng.

Hessian ở đây **không phụ thuộc $w$**, tức độ cong chỗ nào cũng như nhau, vì $f$
là hàm bậc hai. Mặt cong này là một cái bát parabol hoàn hảo, không có chỗ lồi
lõm bất thường, và phần sau khai thác tính chất đó nhiều lần.

### Nghiệm đóng

Cho gradient bằng 0 rồi giải ra:

$$
w^* = \Bigl( \tfrac{1}{n} X^\top X + \lambda I \Bigr)^{-1} \Bigl( \tfrac{1}{n} X^\top y \Bigr)
$$

Tức là ta biết trước đáp án, và câu hỏi tự nhiên là đã biết đáp án rồi thì chạy
thuật toán lặp làm gì.

Nhóm chọn Ridge cho bài tập này chính vì lẽ đó. Biết $f^* = 0.0237430$ chính xác
thì vẽ được sai số tuyệt đối $f(w_k) - f^*$ trên thang logarit. Nếu không có
$f^*$, ta chỉ so được với giá trị tốt nhất từng thấy, và mọi đường sẽ cắm xuống
vô cực ở cuối biểu đồ, che mất hành vi thật. Có $f^*$ thì nhìn thấy
rõ ba thứ: đường thẳng hội tụ tuyến tính, độ dốc gấp của phương pháp tăng tốc, và
cái đuôi nằm ngang của SGD.

---

## 3. Ba con số quyết định tất cả: $L$, $\mu$, $\kappa$

Hình dung hàm mục tiêu là một cái bát. Ta thả một hòn bi ở miệng bát và muốn nó
xuống đáy.

- $L$ là độ cong lớn nhất, hướng dốc nhất của bát.
- $\mu$ là độ cong nhỏ nhất, hướng thoải nhất.
- $\kappa = L/\mu$ là **độ méo** của bát.

$\kappa = 1$ nghĩa là bát tròn xoay hoàn hảo, thả bi phát xuống đáy luôn.
$\kappa$ lớn nghĩa là bát bị bóp dẹt thành cái máng dài. Hòn bi lăn xuống đáy
máng rất nhanh theo chiều ngang, rồi bò dọc theo máng chậm hơn nhiều bậc. Toàn bộ
khó khăn của các phương pháp bậc một nằm ở chỗ này.

Giá trị thực tế của bài:

| Ký hiệu | Ý nghĩa | Giá trị |
| --- | --- | --- |
| $\lambda$ | hệ số hiệu chỉnh | 0,01 |
| $L$ | độ cong lớn nhất | 2,68309 |
| $\mu$ | độ cong nhỏ nhất | 0,01 |
| $\kappa$ | số điều kiện | 268,309 |
| $f^*$ | giá trị tối ưu | 0,0237430 |

### Quan hệ $\mu = \lambda$

Ma trận Gram $\frac{1}{n} X^\top X$ có hạng 273 trên 280, tức có 7 hướng hoàn
toàn suy biến. Và ngay cả trị riêng đầu tiên sau 7 hướng đó cũng chỉ cỡ
$2 \cdot 10^{-8}$, nhỏ hơn $L$ tới tám bậc.

Dữ liệu tự nó có những hướng phẳng lì, không cong chút nào. Cái bát nếu chỉ do dữ
liệu tạo ra thì có một mảng đáy phẳng, hòn bi lăn tới đó là đứng yên, và thứ duy
nhất tạo ra độ cong ở các hướng ấy là vế phạt $\lambda$. Cho nên

$$
\mu = \lambda, \qquad \kappa = \frac{L}{\lambda}
$$

Hệ quả cần thuộc lòng: số điều kiện của bài toán này do $\lambda$ quyết định hoàn
toàn, dữ liệu không có tiếng nói gì. Chương về $\lambda$ vì thế quan trọng hơn
mức kế hoạch dự kiến. Kết luận trên đảo chiều nếu mọi cột độc lập tuyến tính và
trị riêng nhỏ nhất cùng bậc với $\lambda$, khi đó $\mu$ quay về do dữ liệu quyết
định.

---

## 4. Hai quyết định chuẩn bị dữ liệu ảnh hưởng thẳng tới tối ưu hóa

Phần chuẩn bị dữ liệu thường bị coi là công việc phụ, nhưng hai quyết định dưới
đây đổi thẳng số điều kiện của bài toán.

### 4.1. Chuẩn hóa cột: $\kappa$ giảm 173 lần

Cùng dữ liệu, cùng $\lambda$:

| Trường hợp | $\kappa$ | GD chạy 500 vòng đạt được |
| --- | --- | --- |
| Có chuẩn hóa cột | 268,3 | chạm giới hạn số học $10^{-16}$ sau ~290 vòng |
| Giữ thang đo gốc | 46.540 | mới xuống $10^{-2}$ |

Cách nói cho dễ hình dung: cột giá gốc có giá trị cỡ $10^5$, cột số inch màn hình
cỡ 6. Chưa chuẩn hóa thì cái bát bị kéo dài theo hướng giá gốc thành cái máng dài
hàng chục nghìn lần. Chuẩn hóa là ép mọi cột về cùng một thang, tức là **nắn cái
máng lại thành cái bát**.

Chi phí mỗi vòng lặp không đổi, đều khoảng 42 ms, nên biểu đồ theo thời gian chỉ
là biểu đồ theo vòng lặp vẽ lại trên thang khác. Bản chuẩn hóa xuống đáy trong 12
giây, bản kia hơn 20 giây vẫn dừng ở $10^{-2}$.

Chuẩn hóa dữ liệu vì thế không phải thói quen tiền xử lý cho đẹp, mà là can thiệp
trực tiếp lên số điều kiện, tức lên việc bài toán có giải nổi trong thời gian hợp
lý hay không. Lợi ích ấy chỉ dành cho phương pháp bậc một, vì Newton vẫn hội tụ
sau đúng một vòng lặp bất kể $\kappa$ bằng bao nhiêu.

### 4.2. Nhân biến tương tác trên cột đã chuẩn hóa

Trong 280 thuộc tính có 210 cột là tích từng cặp của các cột định lượng. Nếu nhân
trực tiếp trên thang gốc, lấy `original_price` cỡ $10^5$ nhân với
`screen_size_inches` cỡ 6, thì được một cột cỡ $10^6$, một mình nó kéo trị riêng
lớn nhất lên: $L = 17.14$ và 48 trị riêng dưới $10^{-6}$.

Nhân trên cột đã chuẩn hóa thì $L = 2.68$ và chỉ còn 7 trị riêng dưới $10^{-6}$,
đúng bằng mức của khối cột gốc. Số điều kiện tốt lên 6,4 lần, mà **không gian cột
sinh ra thì y hệt nhau**, tức mô hình không mạnh lên hay yếu đi, chỉ là cách biểu
diễn khác.

### 4.3. Chọn $\lambda$: vì sao không lấy điểm cực tiểu của cross-validation

Đường cong cross-validation 5 fold phẳng lì từ $\lambda = 10^{-6}$ đến $10^{-3}$,
sai số giống nhau tới bốn chữ số thập phân. Trong khi đó $\kappa$ giữa hai đầu
khoảng đó chênh nhau 1000 lần.

Lấy điểm cực tiểu của một đường phẳng như vậy là lấy nhiễu của phép chia fold.
Nhóm dùng **quy tắc một sai số chuẩn**: chọn $\lambda$ lớn nhất mà sai số của nó
vẫn nằm trong một sai số chuẩn của giá trị tốt nhất.

| | $\lambda$ | $\kappa$ | RMSE test |
| --- | --- | --- | --- |
| Điểm cực tiểu | $10^{-4}$ | 26.831 | 0,20579 |
| Quy tắc 1-SE (đã chọn) | 0,01 | 268,3 | 0,20601 |

Đổi 0,1% RMSE lấy $\kappa$ nhỏ đi 100 lần. Đánh đổi này có lợi **vì** đường cong
phẳng và **vì** ta giải bằng phương pháp lặp. Nếu đường cong có cực tiểu rõ thì
mất 0,1% là mất thật; nếu dùng phương pháp bậc hai thì $\kappa$ không chi phối số
vòng lặp nên khoản được cũng biến mất.

Mục này cũng là chỗ phân biệt hai bài toán khác nhau:

- Chọn $\lambda$ là bài toán **học máy**, làm một lần lúc chuẩn bị.
- Cực tiểu hóa $f$ với $\lambda$ đã cho là bài toán **tối ưu hóa**, và đó mới là
  nội dung môn học.

---

## 5. Gradient Descent, thuật toán nền

### Ý tưởng

Đứng trên sườn dốc trong sương mù, không nhìn thấy đáy. Cách duy nhất là sờ xem
chỗ nào dốc xuống rồi bước một bước theo hướng đó. Lặp lại.

$$
w_{k+1} = w_k - t \nabla f(w_k)
$$

Thuật toán chỉ có một tham số là $t$, độ dài bước. Bước ngắn quá thì rất lâu mới
tới đáy, còn bước dài quá thì vọt qua đáy sang sườn bên kia, có khi lên cao hơn
cả chỗ vừa đứng.

### Ranh giới $2/L$

Lý thuyết nói hội tụ khi $0 < t < 2/L$. Ở bài này $2/L = 0.745$. Nhóm cố ý cho
chạy $t = 2.1/L$ để xem chuyện gì xảy ra.

| Độ dài bước | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | Kết cục |
| --- | --- | --- | --- |
| $2.1/L$ | không đạt | | **phân kỳ**, $f$ vọt lên $5.4 \cdot 10^{11}$ |
| $2/L$ | 20 | 0,858 | kẹt ở $4.9 \cdot 10^{-8}$ |
| $1.9/L$ | 20 | 0,844 | hội tụ |
| $2/(L+\mu)$ | 20 | 0,773 | hội tụ, nhanh nhất |
| $1/L$ | 40 | 1,650 | hội tụ |
| $0.5/L$ | 80 | 3,013 | hội tụ |
| $0.1/L$ | 380 | 15,926 | hội tụ, chậm |
| $0.01/L$ | không đạt | | 500 vòng mới xuống $3.7 \cdot 10^{-4}$ |

Cơ chế đằng sau ngưỡng ấy nằm ở từng hướng riêng, và hội đồng hay hỏi chỗ này.
Sau mỗi vòng, sai số theo hướng riêng ứng với trị riêng $\lambda_i$ của Hessian
nhân với $|1 - t\lambda_i|$, nên muốn nó tắt dần thì hệ số này phải nhỏ hơn 1. Hướng dốc nhất có $\lambda_i = L$, nên khi
$t$ vượt $2/L$ thì $|1 - tL| > 1$ và thành phần đó **nhân lên** mỗi vòng thay vì
tắt đi. Đúng tại ngưỡng $t = 2/L$, hệ số bằng đúng 1, tức thành phần đó đứng im
mãi mãi, và đó là lý do dòng $2/L$ kẹt ở $4.9 \cdot 10^{-8}$ thay vì đi tiếp.

Chênh 10% quanh ngưỡng đủ để lật từ hội tụ sang phân kỳ, nên khi trình bày thì
chiếu `gd_fixed_iter` và chỉ thẳng vào đường đi lên.

Trong vùng an toàn, chia đôi $t$ thì số vòng lặp nhân đôi, thấy rõ ở dãy 20, 40,
80 ứng với $1.9/L$, $1/L$, $0.5/L$. Vậy nên chọn $t$ càng sát $2/L$ càng tốt. Giá trị lý thuyết $2/(L+\mu)$ tuy tốt nhất nhưng chỉ hơn $1.9/L$ có
0,07 giây, vì $\mu \ll L$ nên $2/(L+\mu) \approx 2/L$.

Lưu ý quy luật tỉ lệ nghịch chỉ đọc được khi ngân sách vòng lặp còn đủ:
$t = 0.01/L$ chạy hết 500 vòng vẫn chưa tới ngưỡng nên không cho ta con số để
đặt vào dãy.

---

## 6. Backtracking: khi không biết $L$

### Ý tưởng

GD bước cố định đòi ta biết $L$ trước. Nếu không tính được thì sao? Thì thử: bắt
đầu bằng một bước dài, nếu thấy không giảm đủ thì rút ngắn lại, cứ nhân với
$\beta < 1$ tới khi chấp nhận được.

Điều kiện chấp nhận gọi là điều kiện Armijo:

$$
f(w - t\nabla f(w)) \le f(w) - \alpha t \|\nabla f(w)\|_2^2
$$

Dịch ra tiếng người: "bước này phải giảm được ít nhất $\alpha$ phần của mức giảm
mà độ dốc hứa hẹn". $\alpha$ là mức độ khó tính, $\beta$ là mức độ rút ngắn mỗi
lần thử.

### Vai trò của $\beta$ và của $\alpha$

| | $\alpha = 0.1$ | $\alpha = 0.3$ | $\alpha = 0.5$ | Vòng lặp đạt $10^{-6}$ |
| --- | --- | --- | --- | --- |
| $\beta = 0.5$ | 1,39 | 1,39 | 1,39 | 15 |
| $\beta = 0.8$ | 2,46 | 2,47 | 2,37 | 20 |
| $\beta = 0.9$ | 4,12 | 4,15 | 3,96 | 20 |

Ba cột giữa là **số lần đánh giá hàm mục tiêu trên mỗi vòng lặp**, đo trong 100
vòng đầu.

Đọc ngang: đổi $\alpha$ gần như không đổi gì, chênh nhiều nhất 4,8%. Lý do là với
hàm bậc hai, điều kiện Armijo chấp nhận $t$ khi
$t \le 2(1-\alpha)\|g\|^2/(g^\top H g)$, nên đổi $\alpha$ trong khoảng vừa phải
chỉ dịch ngưỡng đi chút xíu, thường không đủ để đổi số lần rút ngắn. Kết luận này
sẽ sai nếu $\alpha$ tiến sát 1, vì khi đó ngưỡng co về 0.

Đọc dọc: $\beta$ quyết định chi phí. $\beta = 0.5$ tốn 1,39 lần đánh giá hàm
mỗi vòng, $\beta = 0.9$ tốn 4,1 lần, vì $\beta$ càng gần 1 thì mỗi lần rút ngắn
càng ít nên phải thử nhiều lần hơn.

### Vì sao đề bài bắt vẽ hai biểu đồ

Hai cấu hình $\beta = 0.8$ và $\beta = 0.9$ gần trùng nhau trên trục vòng lặp
nhưng tách hẳn nhau trên trục thời gian, vì chênh lệch nằm ở số lần đánh giá hàm
bên trong mỗi vòng lặp, mà trục vòng lặp không nhìn thấy chi phí đó. Cả bài không
có ví dụ nào sạch hơn ví dụ này.

Backtracking mất 0,852 giây so với 0,773 giây của bước cố định tốt nhất, tức chậm
hơn 10%, đổi lại nó không cần biết $L$. Bài này $L$ tính dễ nên bước cố định hợp
lý hơn, nhưng nhận xét ấy đảo chiều ngay khi $d$ lớn tới mức không tính nổi trị
riêng lớn nhất của ma trận Gram.

---

## 7. SGD: nhanh mà không chính xác

### Ý tưởng

Mỗi vòng GD phải quét cả 160.000 dòng dữ liệu để tính một gradient, và chi phí ấy
rất lớn. SGD chỉ lấy ngẫu nhiên $B$ dòng, tính gradient trên nhúm đó rồi bước
luôn, nên hướng đi lệch lạc nhưng số bước đi được trong cùng thời gian nhiều hơn
hẳn.

Đặt cạnh nhau thì GD là người cẩn thận, đo đạc kỹ rồi mới bước và bước nào cũng
đúng hướng, còn SGD là người bước xiêu vẹo nhưng chân không nghỉ.

### Bẫy thứ nhất, độ dài bước theo kích thước lô

$L$ của toàn bộ hàm là 2,683. Nhưng $L$ của **một mẫu đơn lẻ tệ nhất**, tức
$\max_i \|x_i\|^2 + \lambda$, đo được là **3109,7**, lớn hơn 1159 lần.

Nghĩa là có những dòng dữ liệu mà riêng nó tạo ra một cái bát dốc đứng gấp nghìn
lần cái bát trung bình. Một bước an toàn cho gradient đầy đủ sẽ làm SGD với lô
kích thước 1 phân kỳ.

Thí nghiệm đối chứng chứng minh điều đó: dùng chung $\eta = 0.1/L = 0.0373$
cho mọi kích thước lô thì lô $B = 1$ phân kỳ thành `NaN` ngay trong lượt duyệt dữ
liệu đầu tiên, tức epoch đầu tiên,
còn $B = 1024$ vẫn về được $1.1 \cdot 10^{-4}$.

Nhóm xử lý bằng cách cho mỗi kích thước lô một độ dài bước riêng,
$\eta_0 = 1/L_B$ với

$$
L_B = L + \frac{L_{\max} - L}{B}
$$

Công thức này nội suy giữa hai đầu: $B = 1$ thì $L_B \approx L_{\max}$, $B$ lớn
thì $L_B \to L$.

### Bẫy thứ hai, bước hằng không đưa SGD tới đích

Kết quả dưới đây không phải lỗi cài đặt, dù nhìn qua rất giống.

Với $\eta$ giữ nguyên, SGD đi vào một vùng lân cận quanh $w^*$ có bán kính cỡ
$\mathcal{O}(\eta \sigma^2 / \mu)$ rồi dao động quanh đó mãi. Trên biểu đồ thang
logarit nó hiện ra thành **một đường nằm ngang**.

Gradient của một lô là gradient thật cộng nhiễu, mà càng tới gần đáy thì gradient
thật càng nhỏ trong khi nhiễu không nhỏ đi. Tới lúc nhiễu lấn át tín hiệu, thuật
toán chỉ còn đi lung tung quanh đáy, nên muốn nó đứng lại thì phải hạ dần $\eta$.

Nhóm viết hẳn một phép thử tự động bắt buộc hành vi này phải xảy ra.

### Bốn quy tắc chọn $\eta_k$, cùng $B = 64$, cùng 30 lượt duyệt

| Quy tắc | Sai số cuối |
| --- | --- |
| $\eta_k = \eta_0$ (hằng) | $9.58 \cdot 10^{-4}$ |
| $\eta_k = \eta_0/\sqrt{k+1}$ | $1.46 \cdot 10^{-4}$ |
| $\eta_k = \eta_0/(1+0.5k)$ | $2.90 \cdot 10^{-5}$ |
| $\eta_k = \eta_0 \cdot 2^{-\lfloor k/5 \rfloor}$ (bậc thang) | $5.80 \cdot 10^{-6}$ |

Giảm càng nhanh thì kết quả càng tốt, vì bán kính lân cận co lại theo $\eta_k$.
Quy tắc bậc thang hơn bước hằng **165 lần**. Và cả bốn quy tắc tốn thời gian như
nhau, khoảng 2,4 giây, nên ở đây **không có đánh đổi nào**, bước hằng đơn thuần
kém hơn.

Bốn quy tắc này là công thức tất định theo $k$, không đọc gì từ giá trị hàm mục
tiêu, nên chúng không phải một dạng line search.

### Vì sao SGD không có biến thể backtracking

Hội đồng gần như chắc chắn sẽ hỏi, và có hai lý do để trả lời:

1. **Quá đắt.** Điều kiện Armijo cần tính $f$ trên toàn bộ tập huấn luyện ở mỗi
   lần thử bước. Với $B = 64$, một lượt duyệt có 2500 lần cập nhật. Lấy 39 ms cho
   một lần quét toàn bộ dữ liệu thì riêng line search cần khoảng 98 giây cho một
   lượt duyệt, trong khi cả lượt duyệt thật chỉ mất 0,082 giây.
2. **Không bảo đảm gì.** Áp Armijo lên gradient của lô chỉ bảo đảm giảm cho hàm
   mất mát của lô đó, không bảo đảm cho $f$. Một bước được chấp nhận trên lô này
   vẫn có thể làm $f$ tăng.

### Kích thước lô và thời gian chạy

Khi mỗi lô có bước riêng, cả năm cấu hình đều về mức $10^{-3}$ sau 30 lượt duyệt,
nhưng thời gian chênh 14 lần: 27,4 giây với $B = 1$ so với 1,9 giây với
$B = 1024$.

Chênh lệch này đến từ **chi phí điều phối của Python**, không từ số phép tính:
$B = 1$ thực hiện 160.000 lần cập nhật mỗi lượt duyệt mà mỗi lần chỉ xử lý một
dòng. Nếu cài bằng ngôn ngữ biên dịch, hoặc nếu $d$ lớn hơn nhiều để phần số học
lấn át, tỉ lệ 14 lần này sẽ co lại đáng kể. Nhóm mới dự đoán như vậy chứ chưa đo.

### Kết luận về SGD

Ngay cả quy tắc tốt nhất cũng chỉ xuống $5.80 \cdot 10^{-6}$ sau 30 lượt duyệt,
**chưa chạm ngưỡng $10^{-6}$**, trong khi GD chỉ cần 20 vòng để đạt đúng ngưỡng
ấy. Mà 30 lượt duyệt xấp xỉ chi phí của 30 vòng gradient toàn phần.

Trên bài này SGD vì thế kém hơn các phương pháp còn lại. Nó chỉ đáng dùng khi một
nghiệm tạm với chi phí thấp là đủ, hoặc khi $n$ lớn tới mức quét toàn bộ dữ liệu
một lần đã quá tốn kém.

---

## 8. Accelerated Gradient Descent: thêm quán tính

### Ý tưởng

GD như hòn bi lăn trong máng dính đầy mật, mỗi bước chỉ nghe theo độ dốc tại chỗ.
AGD cho hòn bi có **quán tính**: nó nhớ hướng vừa đi và tiếp tục lao theo hướng
đó một chút.

$$
y_k = w_k + \beta_k (w_k - w_{k-1}), \qquad w_{k+1} = y_k - t \nabla f(y_k)
$$

Nesterov khác momentum thường ở chỗ nó nhìn trước rồi mới lấy gradient: trượt tới
điểm ngoại suy $y_k$ đã, tính gradient ở đó, rồi mới bước, giống như nhìn xa hơn
một bước trước khi quyết định.

Lợi ích lý thuyết: hệ số co rút từ $1 - \mathcal{O}(1/\kappa)$ lên
$1 - \mathcal{O}(1/\sqrt{\kappa})$. Với $\kappa = 268$ thì $\sqrt{\kappa} \approx 16$,
tức về lý thuyết nhanh hơn cỡ 16 lần ở phần đuôi.

### Kết quả ngược với lý thuyết, và vì sao

| Cấu hình | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | Tổng thời gian (s) |
| --- | --- | --- | --- |
| $\beta$ hằng theo $\kappa$ | 35 | 1,496 | 7,12 |
| $\beta$ hằng, có khởi động lại | 35 | 1,500 | 7,13 |
| $\beta_k = (k-1)/(k+2)$ | 20 | 0,879 | 10,50 |
| $\beta_k = (k-1)/(k+2)$, **có khởi động lại** | 20 | 0,879 | **4,86** |
| backtracking $\alpha = 0.5$, $t_0 = 1$ | 20 | 1,462 | 11,45 |
| backtracking $\alpha = 0.3$, $t_0 = 1/L$ | 35 | 2,031 | 10,41 |
| backtracking $\alpha = 0.3$, $t_0 = 1$ | **không đạt** | | 7,54 |

Công thức $\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ mới là công thức **tối
ưu về lý thuyết** cho hàm lồi mạnh, vậy mà nó chậm hơn công thức tăng dần
$\beta_k = (k-1)/(k+2)$, 35 vòng so với 20 vòng.

Công thức hằng nhắm vào trường hợp xấu nhất trong lớp hàm có $\kappa$ cho trước,
tức bài toán có trị riêng trải đều khắp đoạn $[\mu, L]$. Bài toán này có phổ trị
riêng dồn cụm chứ không trải đều, nên bộ tham số phòng thủ ấy mất lợi thế. Trên
bài toán có phổ trải đều, thứ tự hai công thức trở lại đúng như lý thuyết dự
đoán.

### Khởi động lại, khử dao động ở đuôi

Momentum tăng dần gây dao động tuần hoàn, nhìn thấy rõ trên biểu đồ như những gợn
sóng. Hòn bi có quán tính lao qua đáy rồi bị kéo ngược lại, cứ thế đung đưa.

Cơ chế khởi động lại rất đơn giản: khi phát hiện
$\nabla f(y_k)^\top (w_{k+1} - w_k) > 0$, tức là **ta đang đi ngược hướng dốc**,
thì đặt lại $k = 0$, xả hết quán tính, bắt đầu tích lại từ đầu.

Cơ chế này không đổi số vòng lặp để đạt $10^{-6}$, nhưng giảm hơn một nửa tổng
thời gian chạy tới giới hạn số học, từ 10,5 xuống 4,86 giây. Lợi ích nằm trọn ở phần
đuôi, và trên biểu đồ hai đường trùng nhau cho tới ngưỡng $10^{-6}$ rồi mới tách.

Ghép khởi động lại vào công thức $\beta$ hằng thì gần như không đổi gì (7,12 so với
7,13 giây), đúng như dự đoán, vì công thức hằng không tạo dao động tuần hoàn để
mà khử.

### Bẫy khi ghép backtracking với momentum

Với $\alpha = 0.3$ và $t_0 = 1$, phương pháp **không hội tụ**, kẹt ở
$5.5 \cdot 10^{-5}$.

Line search ở đây chạy tại điểm ngoại suy $y_k$ chứ không tại $w_k$, nên nó chỉ
bảo đảm $f(w_{k+1}) < f(y_k)$ và không bảo đảm tính đơn điệu theo $w$. Điều kiện
Armijo lại chỉ chặn độ dài bước dọc theo hướng gradient hiện tại, nên khi
gradient nằm theo hướng có độ cong nhỏ, nó chấp nhận $t$ lớn tới
$\|g\|^2/(g^\top H g)$, có thể lên tới $1/\mu = 100$, trong khi ngưỡng ổn định
toàn cục chỉ là $2/L = 0.745$. Nói cách khác, một hướng thoải đánh lừa line
search, và nó cho qua một bước dài gấp 134 lần mức an toàn.

Hai cách sửa, cả hai đều hiệu quả:

1. Lấy $\alpha = 1/2$, khi đó điều kiện Armijo trở thành đúng bổ đề giảm
   $f(y - t\nabla f(y)) \le f(y) - \frac{t}{2}\|\nabla f(y)\|^2$, tức chính bất
   đẳng thức mà chứng minh hội tụ của AGD dựa vào.
2. Khởi tạo $t_0 = 1/L$, khi đó line search chỉ có thể rút ngắn từ một giá trị
   vốn đã an toàn.

### Lỗi nhóm đã mắc, nên kể khi thuyết trình

Công thức $\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ **chỉ đúng khi
$t = 1/L$**. Nó là trường hợp riêng của công thức tổng quát

$$
\beta = \frac{1 - \sqrt{t\mu}}{1 + \sqrt{t\mu}}
$$

Lần cài đặt đầu, nhóm để line search chọn $t$ nhưng vẫn tính momentum theo $1/L$,
và hàm mục tiêu bùng lên $1.47 \cdot 10^4$. Chương trình chạy trót lọt, không
báo lỗi gì, chỉ trả về một con số trông hợp lý, nên chỉ có đối chiếu với mốc giải
tích mới bắt được loại lỗi này.

---

## 9. Newton và các phương pháp bậc hai

### Ý tưởng

GD chỉ biết hướng dốc. Newton biết cả **độ cong**, nên nó không mò từng bước mà
nhảy thẳng tới đáy của xấp xỉ bậc hai tại chỗ đang đứng:

$$
\nabla^2 f(w_k) p_k = -\nabla f(w_k), \qquad w_{k+1} = w_k + t p_k
$$

### Kết quả: đúng một vòng lặp

Vì $f$ vốn là hàm bậc hai, xấp xỉ bậc hai của nó chính là nó, nên với $w_0 = 0$:

$$
w_1 = w_0 - (\nabla^2 f)^{-1} \nabla f(w_0) = \Bigl( \tfrac{1}{n}X^\top X + \lambda I \Bigr)^{-1} \Bigl( \tfrac{1}{n}X^\top y \Bigr) = w^*
$$

Nói cách khác, bước Newton chính là nghiệm đóng, và Newton ở đây tương đương
việc giải trực tiếp hệ phương trình chuẩn tắc.

| Biến thể | Số vòng lặp | Thời gian (s) |
| --- | --- | --- |
| Newton $t = 1$, phân rã lại mỗi vòng | 1 | 0,078 |
| Newton $t = 1$, dùng lại phân rã | 1 | 0,076 |
| Newton damped, backtracking | 1 | 0,089 |
| Newton-CG, `cg_tol` $= 10^{-2}$ | 5 | 2,07 |
| Newton-CG, `cg_tol` $= 10^{-6}$ | 2 | 1,83 |

Ba dòng đầu gần như không khác nhau, và cần giải thích đúng lý do: backtracking
chấp nhận ngay $t = 1$ nên chỉ tốn một lần đánh giá hàm; còn phân rã lại hay dùng
lại thì vô nghĩa khi chỉ có **một** vòng lặp, không có lần phân rã thứ hai nào để
tiết kiệm. Hai cặp này chỉ tách nhau trên bài toán phi tuyến.

### Vì sao Cholesky chứ không phải `np.linalg.inv`

Nghịch đảo tường minh kém ổn định về số học, và nó tốn nhiều phép tính hơn so với
phân rã rồi giải thế. Hội đồng hay hỏi chỗ này nên cần nói được cả hai lý do.

### Vì sao Newton-CG lại chậm hơn Newton 25 lần

Kết quả này ngược với trực giác, vì Newton-CG sinh ra để tránh chi phí
$\mathcal{O}(d^3)$. Nhưng ở $d = 280$:

- Cholesky tốn $\mathcal{O}(d^3) \approx 2 \cdot 10^7$ phép tính, một lần duy nhất.
- Mỗi bước CG cần một tích Hessian nhân vector, tốn
  $\mathcal{O}(nd) \approx 4.5 \cdot 10^7$, và cần nhiều bước.

Tức là một phép tính $d^3$ còn rẻ hơn một tích Hessian nhân vector, vì
$n = 160.000$ lớn hơn $d = 280$ rất nhiều. Quan hệ này chỉ đảo khi $d$ lớn tới
mức không lập nổi ma trận $d \times d$ trong bộ nhớ, và đó cũng là tình huống duy
nhất khiến Newton-CG đáng chọn.

### L-BFGS: đứng giữa hai nhóm

L-BFGS không lưu Hessian, chỉ giữ $m$ cặp $(s_k, y_k)$ gần nhất rồi dựng xấp xỉ
nghịch đảo Hessian bằng đệ quy hai vòng, chi phí $\mathcal{O}(md)$ mỗi vòng lặp.

| $m$ | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | Vòng lặp tới giới hạn số học | Đánh giá hàm | Tổng thời gian (s) |
| --- | --- | --- | --- | --- | --- |
| 3 | 8 | 0,448 | 54 | 143 | 3,18 |
| 5 | 8 | 0,448 | 50 | 105 | 2,58 |
| 10 | 8 | 0,448 | 40 | 227 | 3,95 |
| 20 | 8 | 0,449 | 37 | 112 | 2,37 |

Hai quan sát cần giải thích:

- **Chọn $m$ không quan trọng ở giai đoạn đầu.** Cả bốn cấu hình đạt $10^{-6}$
  sau đúng 8 vòng lặp, vì trong 8 vòng đầu số cặp tích lũy được còn ít hơn cả
  $m = 3$, nên bốn cấu hình dựng lên gần như cùng một xấp xỉ.
- **Ở đuôi thì $m$ mới có tác dụng**, từ 54 vòng ($m=3$) xuống 37 vòng ($m=20$),
  đều đặn theo $m$, vì càng nhiều cặp thì xấp xỉ bao được càng nhiều hướng riêng.

Nhưng tổng thời gian **không** giảm đều theo $m$: $m = 10$ tốn 3,95 giây trong
khi $m = 3$ chỉ tốn 3,18 giây dù cần nhiều hơn 14 vòng lặp. Chênh lệch đến từ
line search, vốn tiêu 227 lần đánh giá hàm ở $m = 10$ so với 105 lần ở $m = 5$.
Với $d = 280$ thì chi phí $\mathcal{O}(md)$ nhỏ hơn hẳn một lần đánh giá $f$, nên
line search mới là phần chi phối chứ không phải $m$. Với $d$ hoặc $m$ lớn hơn nhiều
thì $\mathcal{O}(md)$ mới chi phối và thời gian sẽ tăng đều theo $m$.

---

## 10. So sánh tổng hợp

| Thuật toán | Cấu hình tốt nhất | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | Tổng thời gian tới giới hạn số học (s) |
| --- | --- | --- | --- | --- |
| Newton | $t = 1$, Cholesky | 1 | **0,076** | 0,076 |
| L-BFGS | $m = 3$ | 8 | 0,448 | 3,18 |
| GD | $t = 2/(L+\mu)$ | 20 | 0,773 | 44,32 |
| GD | backtracking $\alpha = \beta = 0.5$ | 15 | 0,852 | 29,28 |
| AGD | $\beta_k$ tăng dần, có khởi động lại | 20 | 0,879 | **4,86** |
| SGD | $B = 64$, giảm bậc thang | không đạt | | 2,45 |

Bảng này cho ba kết luận, và khi thuyết trình thì nói đúng ba câu đó.

**Một, Newton nhanh nhất trên bài này.** Toàn bộ lời giải mất 0,076 giây, ít hơn
10 lần thời gian GD cần chỉ để đạt $10^{-6}$, và ít hơn 580 lần thời gian GD cần
để chạm giới hạn số học. Nhưng phải nói kèm điều kiện: hàm mục tiêu là bậc hai nên bước Newton
chính là nghiệm đóng, và khoảng cách này **không nói gì** về bài toán phi tuyến.

**Hai, ở độ chính xác vừa phải thì chọn phương pháp nào cũng như nhau.** GD, GD
backtracking và AGD đều đạt $10^{-6}$ trong 0,77 tới 0,88 giây, chênh dưới 14%,
và ba đường bám sát nhau suốt phần đầu biểu đồ. Vì giai đoạn này do các hướng có
độ cong lớn chi phối, mà mọi phương pháp bậc một đều cắt các hướng ấy nhanh như
nhau.

**Ba, ở độ chính xác cao thì thứ hạng đảo lộn.** Để đi tới giới hạn số học, AGD
mất 4,86 giây còn GD mất 44,3 giây, chậm hơn 9 lần. Lợi thế của gia tốc chỉ hiện
ra ở phần đuôi, nơi $\mu$ mới là thứ chi phối. Đề bài bắt vẽ cả hai biểu đồ chính
vì chỗ này, và bài thuyết trình nên chốt ở đây.

### Đối chiếu với lý thuyết

| Phương pháp | Hệ số co rút quan sát | Cận lý thuyết |
| --- | --- | --- |
| GD, $t = 2/(L+\mu)$ | 0,98520 | 0,98520 |
| AGD, $\beta_k$ tăng dần + khởi động lại | 0,65456 | 0,93895 |

Nhóm đo bằng cách khớp một đường thẳng vào phần tuyến tính ở đuôi của đồ thị
$\log_{10}(f(w_k) - f^*)$ theo $k$.

**GD trùng cận lý thuyết tới năm chữ số thập phân.** Không phải trùng ngẫu nhiên:
ở giai đoạn đuôi, đúng thành phần ứng với trị riêng nhỏ nhất chi phối sai số, và
tốc độ tắt của thành phần đó chính là $(\kappa-1)/(\kappa+1)$. Cận này vừa đúng
vừa chặt.

**AGD tốt hơn hẳn cận của nó** (0,65 so với 0,94), và điều này không mâu thuẫn,
vì cận là cận **trên** cho trường hợp xấu nhất. Bài toán cụ thể có phổ trị riêng
tập trung nên dễ hơn trường hợp xấu nhất nhiều.

Chênh lệch lớn nhất nằm ở giai đoạn đầu. Để đạt $10^{-6}$, ước lượng từ lý thuyết
cho khoảng $\frac{1}{2}\kappa \ln(10^6) \approx 1850$ vòng lặp, còn thực tế chỉ
20 vòng, nhỏ hơn gần hai bậc. Cận lý thuyết vì thế mô tả rất tốt phần đuôi nhưng
nói rất ít về giai đoạn đầu, nên không dùng nó để lập ngân sách vòng lặp cho bài
toán này được.

---

## 11. So sánh với scikit-learn

### Quy đổi hàm mục tiêu trước khi so sánh

Mỗi thư viện chuẩn hóa hệ số một kiểu. `Ridge` cực tiểu hóa
$\|Xw-y\|_2^2 + \alpha\|w\|_2^2$, tức không có $\frac{1}{2n}$ và không có
$\frac{\lambda}{2}$. Quy đổi: $\alpha = \lambda n = 0.01 \times 160000 = 1600$.

Không quy đổi thì mọi con số so sánh phía sau là vô nghĩa. Nhóm kiểm chứng bằng
số chứ không tin suy luận trên giấy: nghiệm `Ridge` trả về cách nghiệm đóng
$1.06 \cdot 10^{-14}$, chênh lệch hàm mục tiêu bằng đúng 0 trong số học dấu
phẩy động. Nhóm viết phép kiểm tra này thành một khẳng định trong notebook thay
vì một dòng in ra để đọc bằng mắt, vì mọi so sánh phía sau sẽ vô nghĩa nếu nó
thất bại.

| Bộ ước lượng | $f(\hat w)$ | Sai số so với $f^*$ | Thời gian (s) | RMSE test |
| --- | --- | --- | --- | --- |
| `Ridge(solver='auto')` | 0,023743 | 0 | 0,113 | 0,20601 |
| `Ridge(solver='cholesky')` | 0,023743 | 0 | 0,112 | 0,20601 |
| `Ridge(solver='lsqr')` | 0,023743 | $2.35 \cdot 10^{-9}$ | 0,621 | 0,20601 |
| `Ridge(solver='sag')` | 0,023743 | $1.46 \cdot 10^{-8}$ | 3,955 | 0,20601 |
| `SGDRegressor` (mặc định) | 0,024906 | $1.16 \cdot 10^{-3}$ | 1,018 | 0,21202 |
| `LinearRegression` | 0,023778 | $3.48 \cdot 10^{-5}$ | 1,116 | 0,20579 |

Ba điều rút ra:

1. **`Ridge` với solver trực tiếp đạt $f^*$ với sai số cỡ sai số máy trong 0,11
   giây.** Nó giải thẳng chính cái hệ mà một bước Newton giải. Với bài toán có
   nghiệm đóng và $d$ vài trăm, khó có phương pháp lặp nào sánh được.
2. **`SGDRegressor` mặc định dừng cách $f^*$ tận $1.16 \cdot 10^{-3}$**, cao
   gấp 200 lần sai số $5.80 \cdot 10^{-6}$ của SGD tự cài sau khi điều chỉnh
   quy tắc chọn bước. RMSE cũng kém hơn: 0,21202 so với 0,20601. Vì quy tắc chọn
   bước mặc định của nó không tính theo $L_B$ của bài toán, nên rơi đúng vào cái
   lân cận đã mô tả ở mục 7. Cần nói rõ giới hạn: nhận xét này áp cho bộ ước
   lượng đó ở tham số mặc định, không áp cho scikit-learn nói chung.
3. **`LinearRegression` cho RMSE tốt nhất (0,20579)**, vì nó tương ứng
   $\lambda = 0$, mà quy tắc 1-SE ở mục 4.3 đã cố ý chọn mức hiệu chỉnh mạnh hơn
   mức tối ưu về dự báo.

### Kết luận cân bằng

Mục đích tự cài đặt **không phải** để chạy nhanh hơn thư viện. Nó cho hai thứ mà
một lần gọi hàm thư viện không cho: biết tham số mặc định của thư viện có phù hợp
bài toán của mình không, và biết mỗi thuật toán nhanh hay chậm ở giai đoạn nào,
vì sao. Khi bài toán lớn tới mức không giải trực tiếp được, phương pháp lặp mới
lấy lại chỗ đứng.

---

## 12. Vai trò kép của hệ số hiệu chỉnh

| $\lambda$ | $\mu$ | $\kappa$ | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | RMSE test |
| --- | --- | --- | --- | --- | --- |
| 0,001 | 0,001 | 2674 | 40 | 1,756 | 0,20580 |
| 0,01 | 0,01 | 268,3 | 40 | 1,771 | 0,20601 |
| 0,1 | 0,1 | 27,73 | 30 | 1,233 | 0,21916 |

Tăng $\lambda$ lên 10 lần thì $\kappa$ giảm đúng 10 lần, vì $\mu = \lambda$ trên
bài toán này. Nên $\lambda$ có **hai vai trò tách biệt**:

- Vai trò thống kê: chống quá khớp.
- Vai trò tối ưu hóa: cải thiện điều kiện của bài toán.

Hai vai trò ấy nối chủ đề Ridge với nội dung môn học, nên khi nói thì nhấn vào đó.

Nhưng hai vai trò **không cùng chiều**, và mức ảnh hưởng lệch nhau rất xa. Từ
$\lambda = 0.001$ lên $0.1$: $\kappa$ giảm 96 lần, mà số vòng lặp chỉ giảm từ
40 xuống 30, trong khi RMSE xấu đi 6,5%.

Số vòng lặp ít nhạy cảm như vậy vì cùng lý do đã nói ở mục 10, rằng giai đoạn đầu
do các hướng có độ cong lớn chi phối chứ không phải $\kappa$.

Trên bài này, tăng $\lambda$ để cải thiện điều kiện là đánh đổi không có lợi nếu
chỉ cần độ chính xác vừa phải. Nó chỉ đáng giá khi cần độ chính xác cao, nơi phần
đuôi mới tốn kém và $\kappa$ lấy lại vai trò của mình.

---

## 13. Phần kỹ thuật hay bị hỏi

### Kiểm thử tính đúng đắn (46 phép thử)

Bốn phép thử khai thác việc bài toán có sẵn nghiệm đóng và biểu thức giải tích:

1. Gradient giải tích so với sai phân trung tâm, $\varepsilon = 10^{-6}$, sai số
   tương đối dưới $10^{-6}$.
2. Hessian so với sai phân của gradient, cùng ngưỡng.
3. Nghiệm đóng là điểm dừng: $\|\nabla f(w^*)\| < 10^{-12}$.
4. Trên bài toán tổng hợp $n = 100$, $d = 5$, cả mười cấu hình đều cho
   $\|w_{\text{cuối}} - w^*\| < 10^{-6}$.

Ba phép thử về mặt phương pháp luận, tức bắt buộc thuật toán phải có hành vi
"xấu" đúng như lý thuyết dự đoán:

- GD **phải** phân kỳ khi $t > 2/L$.
- SGD bước hằng **phải** dừng cách $f^*$ một khoảng dương.
- Ghi lịch sử dày hay thưa **phải** cho cùng số lần đánh giá hàm, tức chi phí
  theo dõi không được lẫn vào chi phí thuật toán.

Nhóm bắt được hai lỗi thật nhờ đối chiếu với mốc giải tích, và cả hai đều để
chương trình chạy trót lọt rồi trả về con số trông hợp lý. Không có mốc giải tích
thì chúng còn nguyên trong kết quả.

### Đo thời gian cho đúng

- **Dừng đồng hồ trước khi ghi log.** Tính $f(w_k)$ mỗi vòng tốn ngang một vòng
  gradient descent, nên nếu tính cả vào thì mọi biểu đồ theo trục thời gian sai
  gấp đôi. Dùng `time.perf_counter()`, dừng trước khi tính và ghi, chạy lại sau.
- **Chạy warm-up rồi bỏ kết quả**, vì lần gọi numpy đầu tiên có chi phí khởi tạo.
- **Mỗi cấu hình chạy 3 lần, lấy trung vị.**
- Máy đo: Apple M1 Pro, 16 GB, macOS 26.5, Python 3.14.4, NumPy 2.5.1 liên kết
  Accelerate, SciPy 1.18.0, scikit-learn 1.9.0. Seed 0 cho mọi thành phần ngẫu
  nhiên. Đổi máy thì giá trị tuyệt đối đổi nhưng tỉ lệ giữa các thuật toán gần
  như giữ nguyên, nên khi nói thì bám vào tỉ lệ hơn là bám vào số giây.

### Tiêu chí dừng, một chỗ đã sửa so với kế hoạch

Điều kiện dừng ban đầu là $\|\nabla f(w_k)\| \le 10^{-10}\|\nabla f(w_0)\|$,
nhưng không lần chạy nào đạt được ngưỡng đó, vì độ phân giải của $f$ chặn chuẩn
gradient ở khoảng $10^{-9}$. Một lần chạy backtracking vì thế đạt sai số
$10^{-15}$ ở vòng lặp 115 rồi vẫn chạy tiếp tới vòng 250, và 98,7% trong 72 giây
là thời gian chết. Số lần đánh giá hàm mỗi vòng cũng tăng lên 17 tới 36, do line
search thất bại ở mọi bước thử.

Đã thêm tiêu chí dừng theo đình trệ, có loại trừ các lần chạy đang phân kỳ để
phép thử phân kỳ khi $t > 2/L$ vẫn hoạt động.

### Tổ chức mã nguồn

```
src/
├── problem.py        # RidgeProblem: f, grad, hess, closed_form, L, mu
├── line_search.py    # backtracking Armijo
├── first_order.py    # gd, sgd, agd, heavy_ball, adam
├── second_order.py   # newton, newton_cg, lbfgs
├── runner.py         # chạy lưới tham số, lưu JSON
├── plotting.py       # hàm vẽ chuẩn, bảng màu dùng chung
└── baselines.py      # so sánh với sklearn
```

Mọi thuật toán dùng chung một chữ ký hàm và trả về cùng một cấu trúc lịch sử
(`f_hist`, `gnorm_hist`, `time_hist`, `iter_hist`, `nabla_hist`, `status`), nhờ
vậy ghép kết quả để vẽ không bị lệch. Notebook chỉ gọi hàm và vẽ, không định
nghĩa lại thuật toán, để tránh cùng một thuật toán bị sửa ở ba nơi cho ba kết quả
khác nhau.

Kết quả chạy lưu ra `results/raw/` dạng JSON theo từng nhóm thí nghiệm, và
`run_or_load` bỏ qua nhóm nào đã có file. Nhóm thêm cơ chế này sau khi lần chạy
đầu mất 75 phút vì máy ngủ giữa chừng.

---

## 14. Bộ câu hỏi phản biện và cách trả lời

**"Đã có nghiệm đóng rồi thì chạy thuật toán lặp làm gì?"**
Đúng, và báo cáo nói thẳng điều đó: `Ridge` giải trực tiếp trong 0,11 giây, không
phương pháp lặp nào thắng nổi ở quy mô $d = 280$. Mục đích ở đây là có một hàm
mục tiêu mà ta biết trước đáp án chính xác, để đo sai số tuyệt đối và đối chiếu
tốc độ quan sát với cận lý thuyết. Phương pháp lặp lấy lại chỗ đứng khi $d$ lớn
tới mức chi phí $\mathcal{O}(d^3)$ không chấp nhận được.

**"Vì sao AGD với momentum tối ưu lại chậm hơn momentum tăng dần?"**
Vì công thức hằng thiết kế cho trường hợp xấu nhất trong lớp hàm có $\kappa$ cho
trước, tức phổ trị riêng trải đều trên $[\mu, L]$. Bài toán này có phổ dồn cụm
nên bộ tham số phòng thủ mất lợi thế. Cùng lý do đó khiến hệ số co rút quan sát
được của AGD là 0,65 trong khi cận lý thuyết là 0,94.

**"SGD chạy ra đường nằm ngang, có phải cài sai không?"**
Không, đó là hành vi đúng của SGD bước hằng và nhóm có hẳn một phép thử tự động
bắt buộc nó phải xảy ra. Nghiệm dao động trong lân cận bán kính
$\mathcal{O}(\eta\sigma^2/\mu)$, vì gradient của lô có nhiễu không nhỏ đi khi tới
gần đáy. Hạ dần $\eta_k$ thì bán kính co lại, và quy tắc bậc thang cho kết quả
tốt hơn bước hằng 165 lần.

**"Vì sao không dùng backtracking cho SGD?"**
Điều kiện Armijo cần đánh giá $f$ trên toàn bộ dữ liệu ở mỗi lần thử bước: ước
tính khoảng 98 giây cho một lượt duyệt, trong khi cả lượt duyệt thật chỉ mất
0,082 giây. Ngoài ra Armijo trên gradient của lô không bảo đảm gì cho $f$ toàn
cục.

**"Vì sao $\mu$ đúng bằng $\lambda$?"**
Ma trận Gram có hạng 273 trên 280, và trị riêng kế tiếp sau 7 hướng suy biến chỉ
cỡ $2 \cdot 10^{-8}$, nhỏ hơn $L$ tám bậc. Phần đóng góp của dữ liệu vào $\mu$
coi như bằng 0. Điều kiện đảo chiều: nếu mọi cột độc lập tuyến tính và trị riêng
nhỏ nhất cùng bậc với $\lambda$ thì $\mu$ quay lại do dữ liệu quyết định.

**"Lấy mẫu 200.000 dòng có làm hỏng kết quả không?"**
Không, với các kết luận về tốc độ hội tụ, vì chúng chỉ phụ thuộc $L$, $\mu$ và
$\kappa$. Nhưng sẽ ảnh hưởng nếu đại lượng cần đo phụ thuộc trực tiếp vào $n$,
chẳng hạn phương sai gradient của SGD.

**"Vì sao Newton-CG chậm hơn Newton, trong khi nó sinh ra để rẻ hơn?"**
Vì $n \gg d$ ở bài này. Một lần Cholesky tốn $\mathcal{O}(d^3) \approx 2 \cdot 10^7$,
còn mỗi tích Hessian nhân vector tốn $\mathcal{O}(nd) \approx 4.5 \cdot 10^7$,
mà CG cần nhiều tích như vậy. Quan hệ đảo khi $d$ lớn tới mức không lập nổi ma
trận $d \times d$.

**"Số điều kiện $\kappa$ giảm 96 lần mà số vòng lặp chỉ giảm từ 40 xuống 30?"**
Đúng, và đó là kết quả đáng bàn. Cận lý thuyết chi phối phần đuôi chứ không chi
phối giai đoạn đầu, mà giai đoạn đầu do các hướng có độ cong lớn quyết định. Cùng
hiện tượng ấy giải thích vì sao lý thuyết ước lượng 1850 vòng lặp cho $10^{-6}$
trong khi thực tế chỉ cần 20.

**"Nhóm có gặp lỗi gì không?"**
Có, ba lỗi và đều đáng kể. Một, quy tắc nhận diện cột số có đơn vị biến `model_36`
thành số 36, làm mất cột định tính nhiều mức nhất. Hai, momentum tính theo $1/L$
nhưng ghép với bước do line search chọn, làm hàm mục tiêu bùng lên
$1.47 \cdot 10^4$. Ba, tiêu chí dừng theo chuẩn gradient không bao giờ đạt được
vì độ phân giải của $f$ chặn ở $10^{-9}$, khiến 98,7% thời gian một lần chạy là
thời gian chết.

---

## 15. Kịch bản nói 20 phút

| Phút | Nội dung | Hình cần chiếu | Câu phải nói được |
| --- | --- | --- | --- |
| 0–2 | Bài toán và dữ liệu: 1 triệu dòng, dùng 200.000, $d = 280$ | không | Môn này chấm cách tìm $w$, không chấm dự đoán giá |
| 2–3 | Hàm mục tiêu, gradient, Hessian, nghiệm đóng | không | Hessian hằng số nên có nghiệm đóng, nhờ đó biết $f^*$ |
| 3–5 | $L$, $\mu$, $\kappa$; ví von cái bát méo; $\mu = \lambda$ | bảng hằng số | $\kappa$ do $\lambda$ quyết định, dữ liệu không có tiếng nói |
| 5–6 | Chuẩn hóa cột: $\kappa$ từ 46.540 xuống 268,3 | `normalization_iter` | Tiền xử lý là can thiệp lên bài toán tối ưu hóa |
| 6–8 | GD bước cố định, có đường phân kỳ | `gd_fixed_iter` | Chênh 10% qua $2/L$ là lật từ hội tụ sang $5.4 \cdot 10^{11}$ |
| 8–9 | GD backtracking, bảng số lần đánh giá hàm | `gd_backtracking_time` | $\beta$ quyết định chi phí, $\alpha$ gần như không |
| 9–11 | SGD: kích thước lô và $L_{\max}/L = 1159$ | `sgd_batch_time` | Bước phải theo lô, dùng chung bước thì $B=1$ ra `NaN` |
| 11–12 | SGD: bốn quy tắc, đường nằm ngang của bước hằng | `sgd_schedule_epoch` | Bước hằng dừng ở lân cận, đó là lý thuyết chứ không phải lỗi |
| 12–13 | AGD: hai công thức momentum, khởi động lại | `agd_time` | Khởi động lại giảm tổng thời gian từ 10,5 xuống 4,86 giây |
| 13–14 | Newton: một vòng lặp, và vì sao | `newton_time` | Bước Newton chính là nghiệm đóng |
| 14–16 | So sánh tổng hợp, hai biểu đồ | `all_methods_iter`, `all_methods_time` | Thứ hạng đảo giữa hai biểu đồ, đó là cả câu chuyện |
| 16–17 | Lý thuyết so với thực nghiệm | `theory_vs_practice` | GD trùng cận tới 5 chữ số ở đuôi, sai gần 2 bậc ở đầu |
| 17–18 | So sánh sklearn, quy đổi $\alpha = \lambda n$ | `sklearn_time` | Tự cài không để nhanh hơn thư viện, mà để biết vì sao |
| 18–20 | $\lambda$ và kết luận | `lambda_effect_iter` | $\lambda$ có hai vai trò, và chúng không cùng chiều |

### Năm câu nếu chỉ được nói năm câu

1. Hàm mục tiêu là bậc hai lồi mạnh nên có nghiệm đóng, nhờ vậy đo được sai số
   tuyệt đối $f(w_k) - f^*$ chứ không phải sai số tương đối với giá trị tốt nhất
   từng thấy.
2. Toàn bộ tốc độ hội tụ của nhóm bậc một do $\kappa = L/\mu$ quyết định, và trên
   bài này $\kappa = L/\lambda$ vì ma trận Gram suy biến.
3. Ngưỡng $2/L$ là ranh giới thật: chênh 10% là lật từ hội tụ 20 vòng sang phân
   kỳ tới $5.4 \cdot 10^{11}$.
4. Ở độ chính xác vừa phải mọi phương pháp bậc một như nhau, chênh dưới 14%; ở độ
   chính xác cao AGD nhanh hơn GD 9 lần. Đó là lý do phải vẽ cả hai biểu đồ.
5. Newton hội tụ một vòng lặp vì bước Newton chính là nghiệm đóng, nhưng kết luận
   này gắn chặt với dạng bậc hai và không chuyển sang bài toán phi tuyến.
