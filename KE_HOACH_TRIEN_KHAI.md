# Kế hoạch triển khai bài tập môn Tối ưu hóa nâng cao

> **Cho người thực hiện:** kế hoạch này dùng cú pháp checkbox `- [ ]` để theo dõi
> tiến độ. Mỗi việc làm xong thì đánh dấu và commit.

**Mục tiêu:** viết lại báo cáo và slide theo một mạch dẫn dắt duy nhất, nối kết
quả tối ưu hóa với bài toán định giá, và giải thích đầy đủ mọi công thức đang dùng.

**Cách làm:** giữ nguyên toàn bộ số liệu đã có, thêm đúng một thí nghiệm, đổi trật
tự chương để câu trả lời chính xuất hiện sớm, sắp phần giải thích theo ba cơ chế
thay vì theo bảy thuật toán, và cắt ba thuật toán khỏi mạch chính.

**Công nghệ:** Python 3 với numpy, scipy, scikit-learn, matplotlib; LaTeX biên
dịch bằng `latexmk -xelatex`; pytest.

**Spec:** `docs/superpowers/specs/2026-08-17-tai-cau-truc-bao-cao-design.md`

**Chủ đề:** ứng dụng các thuật toán tối ưu hóa bậc một và bậc hai trong bài toán
hồi quy tuyến tính có hiệu chỉnh Ridge để định giá điện thoại đã qua sử dụng.

**Dữ liệu:** [Used Phone Price Prediction Dataset](https://www.kaggle.com/datasets/sharmajicoder/used-phone-price-prediction-dataset) (Kaggle).

## Ràng buộc chung

Mọi việc trong kế hoạch này đều phải giữ các ràng buộc sau.

- Hàm mục tiêu đã chốt và không được đổi. `data/processed/problem_config.json`
  giữ nguyên, không sửa.
- Mã nguồn viết bằng tiếng Anh, không có ngoại lệ. Nội dung cho người đọc viết
  bằng tiếng Việt.
- Đo thời gian bằng `time.perf_counter()`, dừng đồng hồ trước khi tính và ghi log.
- Mỗi so sánh có hai hình: một theo số vòng lặp, một theo thời gian chạy tính bằng
  giây. Trục tung dùng `semilogy` với $f(w_k) - f^*$.
- Mỗi hình lưu ra `results/figures/` ở hai định dạng PDF và PNG, `dpi` từ 150 trở
  lên, luôn dùng `bbox_inches='tight'`.
- Mỗi thuật toán một màu cố định, định nghĩa một lần trong `src/plotting.py`.
- Legend ghi rõ tham số, ví dụ `GD (t = 1/L)`.
- Mọi hình và bảng trong báo cáo có `\caption`, có `\label`, và được `\ref` ít
  nhất một lần.
- Mọi kết luận kèm số liệu hoặc hình. Chưa có bằng chứng thì ghi rõ là dự đoán.
- Cố định seed cho mọi thành phần ngẫu nhiên và ghi seed vào kết quả.

---

## 0. Trạng thái hiện tại

Phần cài đặt và thí nghiệm đã xong. Phần còn lại là viết lại báo cáo và slide.

| Hạng mục | Trạng thái |
| --- | --- |
| Chuẩn bị dữ liệu, chốt $\lambda$, tính $L$, $\mu$, $\kappa$, $f^*$ | Xong |
| Tám cài đặt thuật toán trong `src/` | Xong |
| Kiểm thử tính đúng đắn | Xong, `tests/` xanh |
| Toàn bộ lưới tham số, kết quả ở `results/raw/` | Xong |
| Hình ở `results/figures/` | Xong, cần bổ sung hai hình mới |
| Báo cáo và slide | Cần viết lại toàn bộ |

Lý do viết lại: giáo viên đánh giá bài khó theo dõi, chủ đề không có dẫn dắt, và
phần thuật toán không nói được tác động tới bài toán ứng dụng. Chẩn đoán chi tiết
nằm ở mục 1 của spec.

---

## 1. Bài toán và ký hiệu

Đây là nguồn tham chiếu duy nhất về ký hiệu. Báo cáo, slide và mã nguồn đều theo
mục này.

$n$ là số điểm dữ liệu, $d$ là số thuộc tính sau khi mã hóa, $X \in \mathbb{R}^{n \times d}$
là ma trận thiết kế, $y \in \mathbb{R}^n$ là vector mục tiêu, $w$ là tham số,
$\lambda$ là hệ số hiệu chỉnh, $t$ và $\eta$ là độ dài bước, $L$ là hằng số
Lipschitz của gradient, $\mu$ là hệ số lồi mạnh, $\kappa$ là số điều kiện, $f^*$
là giá trị tối ưu.

Hàm mục tiêu, sau khi chuẩn hóa cột của $X$ và trừ trung bình khỏi $y$ nên hệ số
chặn bằng không:

$$
f(w) = \frac{1}{2n} \left\| Xw - y \right\|_2^2 + \frac{\lambda}{2} \left\| w \right\|_2^2
$$

Hệ số $\frac{1}{2}$ để đạo hàm không kéo theo số 2. Hệ số $\frac{1}{n}$ để $L$
không phụ thuộc cỡ mẫu, nhờ vậy kết luận về độ dài bước còn dùng được khi đổi số
điểm dữ liệu.

Gradient và Hessian:

$$
\nabla f(w) = \frac{1}{n} X^{\top} (Xw - y) + \lambda w,
\qquad
\nabla^2 f(w) = \frac{1}{n} X^{\top} X + \lambda I
$$

Hessian là ma trận hằng, không phụ thuộc $w$. Đây là tính chất khiến phần lớn bài
toán đơn giản đi, kể cả chuyện Newton hội tụ sau một vòng lặp.

Nghiệm đóng và giá trị tối ưu:

$$
w^* = \left( \frac{1}{n} X^{\top} X + \lambda I \right)^{-1} \left( \frac{1}{n} X^{\top} y \right),
\qquad f^* = f(w^*)
$$

Ba hằng số quyết định tốc độ hội tụ:

$$
L = \lambda_{\max}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\mu = \lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\kappa = \frac{L}{\mu}
$$

Cách đọc hình học của $\kappa$: đường mức của $f$ là các ellipsoid, và $\kappa$ là
tỉ số giữa trục dài nhất và trục ngắn nhất. $\kappa$ lớn nghĩa là thung lũng hẹp,
nên gradient descent đi zigzag ngang thung lũng thay vì đi dọc xuống đáy.

Giá trị cụ thể của bài toán này nằm ở mục 2.

---

## 2. Kết quả đã có

### 2.1. Quy mô và hằng số

| Đại lượng | Giá trị |
| --- | --- |
| Số dòng bộ dữ liệu gốc | 1.000.000, 28 cột, không thiếu, không trùng |
| Mẫu dùng cho thí nghiệm | 200.000 dòng, chia 160.000 huấn luyện và 40.000 kiểm tra |
| $d$ sau khi mã hóa | 280 |
| $\lambda$ | 0,01 |
| $L$ | 2,68309 |
| $\mu$ | 0,01, bằng đúng $\lambda$ vì ma trận Gram hạng 273 trên 280 |
| $\kappa$ | 268,3 |
| $L_{\max}$ của mất mát một mẫu | 3109,7, tức lớn hơn $L$ 1159 lần |

### 2.2. Bảng kết quả chính

Nguồn: `results/raw/summary_all_methods.csv`.

| Thuật toán | Cấu hình | Vòng lặp đạt $10^{-6}$ | Thời gian (s) | RMSE test |
| --- | --- | --- | --- | --- |
| Newton | $t = 1$, phân rã một lần | 1 | 0,076 | 0,2060141 |
| L-BFGS | $m = 3$ | 8 | 0,448 | 0,2060141 |
| AGD | $\beta_k = (k-1)/(k+2)$, có restart | 20 | 0,879 | 0,2060141 |
| GD | $t = 2/(L+\mu)$ | 20 | 0,773 | 0,2060141 |
| GD backtracking | $\alpha = 0{,}5$, $\beta = 0{,}5$ | 15 | 0,852 | 0,2060141 |
| SGD bậc thang | $B = 64$ | không đạt | -- | 0,2060386 |
| SGD bước hằng | $B = 256$ | không đạt | -- | 0,2098552 |

Hai quan sát chi phối toàn bộ mạch báo cáo mới:

1. Năm cấu hình đầu cho RMSE trùng nhau tới chữ số thứ bảy, trong khi thời gian
   chênh 580 lần giữa Newton và toàn bộ ngân sách của GD.
2. Chỉ SGD bước hằng mới đổi được chất lượng dự đoán, và đổi theo hướng xấu đi.

### 2.3. Hình đã có

`results/figures/` hiện có 22 hình, mỗi hình một bản PDF và một bản PNG: các nhóm
`gd_fixed`, `gd_backtracking`, `sgd_batch`, `sgd_schedule`, `sgd_common_step`,
`agd`, `newton`, `lbfgs`, `all_methods`, `sklearn`, `lambda_effect`,
`normalization`, `lambda_cv`, `theory_vs_practice`.

Hai hình cần thêm, sinh từ thí nghiệm E1 ở mục 6:

- `rmse_vs_gap`: RMSE trên tập kiểm tra theo $f(w_k) - f^*$.
- `rmse_vs_time`: RMSE trên tập kiểm tra theo thời gian chạy.

---

## 3. Dàn bài báo cáo

Sợi chỉ đỏ, đặt ở cuối chương 1 và nhắc lại ở đầu mỗi chương: *huấn luyện mô hình
định giá là giải một bài toán tối ưu hóa; giải bằng cách nào, và giải tới mức nào
thì dừng?*

Hai quy ước áp cho toàn báo cáo:

1. Trả lời trước, giải thích sau. Câu trả lời chính nằm ở chương 3 và 4.
2. Mỗi chương đóng bằng một câu quy ra ngôn ngữ ứng dụng, tức bằng giây, bằng
   RMSE, hoặc bằng tiền.

| Chương | Câu hỏi chương trả lời |
| --- | --- |
| 1. Bài toán định giá và hàm mục tiêu | Bài toán ứng dụng là gì, đo một lời giải tốt bằng gì, cái gì quyết định độ khó |
| 2. Chuẩn bị dữ liệu | Mỗi quyết định chuẩn bị dữ liệu làm bài toán dễ hay khó đi, đo bằng $\kappa$ |
| 3. Tối ưu tới mức nào là đủ | Sai số $f - f^*$ nhỏ tới đâu thì sai số giá ngừng giảm |
| 4. Thuật toán nào về đích trước | Năm thuật toán ở cấu hình tốt nhất, đặt cạnh thư viện |
| 5. Cài đặt | Năm thuật toán cài thế nào, chữ ký hàm chung, cách đo thời gian |
| 6. Ba cơ chế giải thích mọi chênh lệch | Vì sao các đường nằm ở đó |
| 7. Lý thuyết so với thực nghiệm | Cận lý thuyết đúng chỗ nào, sai chỗ nào |
| 8. Kết luận bền tới đâu | Đổi $\lambda$ tức đổi $\kappa$ thì thứ hạng có đổi không |
| 9. Kết luận | Trả lời câu hỏi ở chương 1, dạng lời khuyên kèm điều kiện đảo chiều |

Phụ lục A kiểm thử tính đúng đắn, B ba thuật toán không đưa vào so sánh chính,
C quy đổi hàm mục tiêu của scikit-learn, D dẫn xuất công thức, E cấu hình và seed.

### 3.1. Thuật toán trong mạch chính

Năm thuật toán: GD với hai biến thể độ dài bước là cố định và backtracking, SGD,
AGD, Newton, L-BFGS. Heavy ball, Newton-CG và Adam chuyển xuống phụ lục B dưới dạng một mục
ngắn nói rõ đã cài và vì sao không đưa vào so sánh. Yêu cầu "áp dụng thêm thuật
toán khác" của đề bài vẫn đủ nhờ L-BFGS.

Không làm mất mát Huber. Bài học về Newton trên hàm bậc hai đã đủ nếu nói thẳng
rằng bước Newton chính là nghiệm đóng, nên Newton ở đây không phải một thuật toán
lặp. Chương 9 dành một đoạn nói kết luận nào sẽ đổi trên hàm phi tuyến.

### 3.2. Chương 6 sắp theo ba cơ chế

Bảy mục khảo sát tham số hiện tại giữ nguyên số liệu nhưng sắp lại thành ba mục.

| Cơ chế | Hiện tượng nó giải thích | Hình dùng |
| --- | --- | --- |
| Độ dài bước so với ngưỡng $2/L$ | GD phân kỳ ở $t = 2.1/L$ với $f$ tăng tới $5{,}4 \cdot 10^{11}$; số vòng lặp tỉ lệ nghịch với $t$ theo dãy 20, 40, 80; chi phí backtracking do $\beta$ quyết định chứ không do $\alpha$, với $\beta = 0{,}5$ tốn 1,39 lần đánh giá hàm mỗi vòng còn $\beta = 0{,}9$ tốn 4,1 lần; momentum phải khớp với bước thực tế, ghép sai làm $f$ bùng lên $1{,}47 \cdot 10^4$ | `gd_fixed_*`, `gd_backtracking_*`, `agd_*` |
| Nhiễu của gradient ngẫu nhiên | $L_{\max}$ lớn hơn $L$ 1159 lần nên độ dài bước phải theo kích thước lô; bước hằng dừng ở một lân cận quanh $w^*$; giảm theo bậc thang hơn bước hằng 165 lần với cùng chi phí | `sgd_batch_*`, `sgd_schedule_*`, `sgd_common_step_*` |
| Chi phí mỗi vòng lặp | Newton cần một vòng nhưng vòng đó tốn $\bigO(nd^2 + d^3)$; L-BFGS đứng giữa hai nhóm; thứ tự các đường đổi khi trục hoành chuyển từ vòng lặp sang giây | `newton_*`, `lbfgs_*`, `all_methods_*` |

Ba lỗi ghi ở mục 12 đưa vào đúng cơ chế tương ứng, không giấu đi: chúng là bằng
chứng thực nghiệm mạnh nhất cho hai cơ chế đầu.

---

## 4. Dàn bài slide

Sáu phần, bám đúng mạch báo cáo. Thời lượng dự kiến 20 phút.

1. Bài toán định giá và cách đo một lời giải tốt
2. Tối ưu tới mức nào là đủ, một hình
3. Thuật toán nào về đích trước, ba hình và một bảng
4. Ba cơ chế giải thích chênh lệch
5. Lý thuyết đúng ở đuôi, sai ở giai đoạn đầu
6. Kết luận dạng lời khuyên kèm điều kiện

Phụ lục slide giữ phần khảo sát tham số chi tiết, ba thuật toán đã cắt, phần kiểm
thử, và quy đổi hàm mục tiêu của thư viện.

---

## 5. Danh mục công thức và cách xử lý

Mỗi công thức trong mạch chính trả lời ba câu hỏi theo thứ tự: **nó nói gì** (đọc
thành lời, ưu tiên cách đọc hình học), **nó từ đâu ra** (dẫn xuất ngắn hoặc nguồn),
**nó dùng vào việc gì ở bài này** (số cụ thể và chương khai thác). Bản cũ phần lớn
chỉ có phần thứ ba, nên công thức rơi từ trên trời xuống.

Mười sáu công thức, gồm mười bốn công thức đã có nhãn cộng cập nhật của GD và đệ
quy hai vòng của L-BFGS hiện chưa đánh nhãn.

**Nhóm 1, định nghĩa bài toán. Chương 1.**

| Công thức | Phần cần bổ sung |
| --- | --- |
| `eq:objective` | Vì sao có hệ số $1/2$ và vì sao chia cho $n$; công thức thể hiện đánh đổi giữa khớp dữ liệu và giữ tham số nhỏ |
| `eq:derivatives` | Hessian là ma trận hằng, không phụ thuộc $w$ |
| `eq:closed-form` | Tồn tại và duy nhất nhờ $\lambda > 0$; hệ quả là $f^*$ biết trước nên sai số đo được là tuyệt đối |

**Nhóm 2, ba con số quyết định tốc độ. Chương 1, dùng lại ở chương 7 và 8.**

| Công thức | Phần cần bổ sung |
| --- | --- |
| `eq:constants` | Với Hessian hằng, hằng số Lipschitz của gradient chính là chuẩn phổ của Hessian; kèm cách đọc hình học của $\kappa$ ở mục 1 |
| `eq:mu-equals-lambda` | Ma trận Gram hạng 273 trên 280, trị riêng kế tiếp cỡ $2 \cdot 10^{-8}$; hệ quả là độ khó bài toán do người làm chọn qua $\lambda$ |

**Nhóm 3, cập nhật của từng thuật toán. Chương 5, bảng tóm tắt đặt ở mục 1.5.**

| Công thức | Thông tin dùng | Chi phí mỗi vòng |
| --- | --- | --- |
| Cập nhật GD | Gradient tại một điểm | $\bigO(nd)$ |
| `eq:sgd` | Gradient trên lô $B$ điểm | $\bigO(Bd)$ |
| `eq:agd` | Gradient tại điểm ngoại suy, cộng một điểm nhớ | $\bigO(nd)$ |
| `eq:newton-one-step` | Cả Hessian | $\bigO(nd^2 + d^3)$ |
| Đệ quy hai vòng của L-BFGS | $m$ cặp hiệu gần nhất | $\bigO(nd + md)$ |

**Nhóm 4, chọn tham số. Chương 6, ba công thức đầu rơi đúng vào ba cơ chế.**

| Công thức | Cơ chế | Phần cần bổ sung |
| --- | --- | --- |
| `eq:armijo` | Độ dài bước | Vì sao đòi giảm ít nhất tỉ lệ $\alpha$ của mức giảm tuyến tính dự đoán thay vì chỉ đòi giảm; $\alpha$ định nghĩa thế nào là giảm đủ còn $\beta$ định chi phí tìm kiếm |
| `eq:momentum-from-step` | Độ dài bước | Vì sao momentum phải tính theo bước mà line search thực sự chọn |
| `eq:minibatch-smoothness` | Nhiễu gradient | Vì sao $L_B$ nội suy giữa $L_{\max}$ và $L$ |
| `eq:schedules` | Nhiễu gradient | Vì sao phương sai gradient không tắt theo $k$ nên phải để $\eta_k$ tắt thay |

**Nhóm 5, bảo đảm lý thuyết. Chương 7.**

| Công thức | Phần cần bổ sung |
| --- | --- |
| `eq:descent-lemma` | Là bất đẳng thức nền của mọi chứng minh hội tụ trong bài; đưa lên sớm thay vì để xuất hiện lần đầu ở trang 45 như một mẹo kỹ thuật |
| `eq:gd-rate` | Đọc thành lời: mỗi vòng lặp sai số nhân với một hệ số cố định nhỏ hơn 1, hệ số đó tiến về 1 khi $\kappa$ lớn |

Thêm bảng ký hiệu một trang ở đầu báo cáo, cột cuối ghi giá trị cụ thể của bài
toán này, lấy từ mục 2.1. Dẫn xuất dài nằm ở phụ lục D.

---

## 6. Giai đoạn A: thay bộ quy tắc văn phong

- [x] Đã xong. Ghi lại ở đây vì mọi việc viết phía sau đều dựa vào kết quả này.

Bản cũ của `docs/van-phong-tieng-viet.md` dài 300 dòng, gồm quy trình ba lượt và
khoảng 20 điều cấm. Nó đã được áp dụng đúng một lần, ở commit `364e7de` viết lại
`report.tex`, và sản phẩm là bản báo cáo bị đánh giá là khô và khó theo dõi. Bộ
đối chứng ở `style/doi-chung/` thì chưa từng chạy: từ khi nó ra đời ở `32f7b0e`,
file văn phong không được sửa lần nào.

Bốn nguyên nhân đã chẩn đoán. Bộ quy tắc thao tác toàn bộ ở cấp câu và cấp đoạn
trong khi lời chê nằm ở cấp tài liệu. Nó chỉ có phần cấm, còn phần sinh là bộ mẫu
thì nằm ở `style/mau/`, một thư mục không tồn tại trên đĩa. Ba điều cấm của nó
chặn đúng những thứ dàn bài mới bắt buộc phải có, gồm câu mở chương nêu câu hỏi và
câu chốt chương quy ra giây hoặc ra tiền. Và giao thức đo ở mục 8 nặng tới mức
không ai chạy.

Nguyên tắc thay thế: **thứ gì máy kiểm được thì nằm ở test và không xuất hiện
trong tài liệu nào; thứ gì máy không kiểm được thì nằm ở vai và ở mẫu.** Không có
loại thứ ba, tức danh sách quy tắc phải nhớ và phải rà bằng mắt, vì đó đúng là
loại duy nhất đã chứng minh là không hiệu quả.

Kết quả:

| Việc | Trạng thái |
| --- | --- |
| `docs/van-phong-tieng-viet.md` viết lại còn 5 mục: vai, sáu đoạn mẫu, bốn lỗi cấu trúc tiếng Anh, ba ràng buộc cứng, và phần máy kiểm | Xong |
| Bộ mẫu lấy từ chính dự án: bốn đoạn từ `report.tex`, hai đoạn từ `docs/giai-thich-de-thuyet-trinh.md` | Xong, thay cho `style/mau/` đã mất |
| `tests/test_style.py`: em dash, dấu thập phân trong văn xuôi, nhất quán thuật ngữ | Xong, sáu phép kiểm xanh trên `report.tex` và `slides.tex` |
| `CLAUDE.md` mục 2 rút còn bốn đoạn | Xong |
| `style/` xóa khỏi cây làm việc, còn trong lịch sử git | Xong |

Quy trình viết từ nay: nạp file văn phong, viết một lượt, chạy `pytest tests/test_style.py`.
Không còn lượt rà riêng.

---

## 7. Giai đoạn B: thí nghiệm E1

Ghi RMSE trên tập kiểm tra tại từng lần ghi lịch sử, cho mọi thuật toán ở cấu hình
tốt nhất. Đây là thí nghiệm mới duy nhất, và chương 3 của báo cáo phụ thuộc vào nó.

**File:**
- Sửa: `src/history.py`, lớp `Recorder` và lớp `OptimizeResult`
- Sửa: `src/problem.py`, lớp `RidgeProblem.__init__`
- Sửa: `src/plotting.py`, thêm hai hàm vẽ
- Sửa: `notebooks/05_comparison_all.ipynb`
- Test: `tests/test_correctness.py`, dùng fixture `problem` đã có ở đó, tức
  `make_synthetic_ridge(n=100, d=5, lam=1e-2, seed=0)`

**Giao diện:**
- Dùng: `baselines.rmse(problem, w, X_test, y_test)` đã có ở `src/baselines.py:146`
- Tạo ra: thuộc tính `RidgeProblem.monitor`, trường `OptimizeResult.metric_hist`,
  khóa `metric_hist` trong JSON, hai hàm `plot_rmse_vs_gap` và `plot_rmse_vs_time`

Thiết kế: gắn hàm đo vào đối tượng `problem` thay vì thêm tham số cho tám hàm
thuật toán. `Recorder` đã nhận `problem` và đã dừng đồng hồ trong `record`, nên
chỉ cần đọc thuộc tính ở đúng cửa sổ đã dừng đồng hồ. Không hàm thuật toán nào
phải đổi chữ ký.

- [ ] **Bước 1: viết test hỏng**

Thêm vào `tests/test_correctness.py`. Các import cần dùng đã có sẵn ở đầu file đó.

```python
def test_recorder_collects_monitor_values(problem: RidgeProblem) -> None:
    """The monitor runs once per recorded point, inside the paused window."""
    problem.monitor = lambda w: float(np.linalg.norm(w))

    result = gradient_descent(
        problem,
        w0=np.zeros(problem.d),
        max_iter=10,
        record_every=1,
        patience=None,
    )

    assert len(result.metric_hist) == len(result.f_hist)
    assert result.metric_hist[0] == 0.0
    assert result.metric_hist[-1] > 0.0
```

- [ ] **Bước 2: chạy test để xác nhận nó hỏng**

Chạy: `pytest tests/test_correctness.py::test_recorder_collects_monitor_values -v`
Kết quả mong đợi: FAIL với `AttributeError`, vì `OptimizeResult` chưa có
`metric_hist`.

- [ ] **Bước 3: thêm thuộc tính `monitor` cho `RidgeProblem`**

Thêm `from collections.abc import Callable` vào đầu `src/problem.py`, rồi trong
`RidgeProblem.__init__`, cạnh phần khai báo bộ đếm:

```python
        # Optional callable evaluated by the Recorder inside its paused
        # window, so its cost stays out of the reported running time.
        self.monitor: Callable[[np.ndarray], float] | None = None
```

- [ ] **Bước 4: thêm `metric_hist` cho `OptimizeResult`**

Thêm trường cạnh `access_hist`:

```python
    metric_hist: list[float] = field(default_factory=list)
```

Thêm `"metric_hist": self.metric_hist` vào `to_dict`, và
`metric_hist=list(payload.get("metric_hist", []))` vào `from_dict`. Dùng `get` với
mặc định rỗng để các file JSON cũ vẫn đọc được.

- [ ] **Bước 5: gọi monitor trong `Recorder.record`**

Trong `src/history.py`, thêm `self.metric_hist: list[float] = []` vào `__init__`,
rồi trong `record`, ngay sau khi khôi phục bộ đếm và trước phần `append`:

```python
        monitor = getattr(self.problem, "monitor", None)
        if monitor is not None:
            self.metric_hist.append(float(monitor(w)))
```

Truyền `metric_hist=self.metric_hist` trong `finish`.

- [ ] **Bước 6: chạy test để xác nhận nó xanh**

Chạy: `pytest tests/ -v`
Kết quả mong đợi: PASS, và các test cũ vẫn xanh. Các file JSON cũ trong
`results/raw/` vẫn đọc được nhờ `payload.get("metric_hist", [])` ở bước 4.

- [ ] **Bước 7: commit**

```bash
git add src/history.py src/problem.py tests/test_correctness.py
git commit -m "feat: record an optional per-iteration metric without timing it"
```

- [ ] **Bước 8: chạy lại các nhóm thí nghiệm để sinh trường mới**

Trong `notebooks/05_comparison_all.ipynb`, trước khi chạy lưới:

```python
X_test = np.load(PROCESSED / "X_test.npy")
y_test = np.load(PROCESSED / "y_test.npy")
problem.monitor = lambda w: rmse(problem, w, X_test, y_test)
```

Xóa các file JSON tương ứng trong `results/raw/` để `run_or_load` chạy lại thật,
giữ nguyên seed đã ghi trong kết quả cũ. Xác nhận cột `time_to_1e-06_s` trong
`summary_all_methods.csv` không đổi quá 20% so với bản cũ; lệch nhiều hơn nghĩa là
monitor đang bị tính vào thời gian.

- [ ] **Bước 9: vẽ hai hình mới**

Thêm vào `src/plotting.py` hai hàm dùng bảng màu chung đã có:

  - `plot_rmse_vs_gap(results, f_star)`, trục hoành $f(w_k) - f^*$ thang log đảo
    chiều, trục tung RMSE, mỗi thuật toán một đường.
  - `plot_rmse_vs_time(results)`, trục hoành giây, trục tung RMSE.

Lưu ra `results/figures/rmse_vs_gap.{pdf,png}` và `rmse_vs_time.{pdf,png}`.

- [ ] **Bước 10: đọc kết quả và ghi ngưỡng $\eps_{\text{app}}$**

Tìm giá trị $f - f^*$ mà từ đó RMSE ngừng cải thiện. Ghi con số này vào
`results/raw/` dưới dạng một khóa trong file JSON tóm tắt, vì chương 3 và chương 4
đều dùng nó.

**Dự đoán chưa kiểm chứng:** RMSE bão hòa từ khoảng $10^{-4}$ trở xuống. Nếu kết
quả cho thấy RMSE tiếp tục giảm tới tận $10^{-8}$ thì kết luận của chương 3 đảo
chiều, và chương phải viết theo hướng ngược lại. Không viết chương 3 trước khi có
số này.

- [ ] **Bước 11: commit**

```bash
git add src/plotting.py notebooks/05_comparison_all.ipynb results/
git commit -m "feat: add test-RMSE curves against optimization gap and wall time"
```

---

## 8. Giai đoạn C: viết lại báo cáo

Mỗi chương là một việc riêng, làm theo thứ tự, commit sau mỗi chương. Với mỗi
chương, quy trình giống nhau:

1. Dựng bảng lập luận bốn cột trước, mỗi đoạn dự kiến một dòng: kết luận, số liệu
   chống lưng, cơ chế, điều kiện đảo chiều. Chưa viết câu văn nào. Dòng nào chưa
   điền được cột cơ chế thì chưa đủ nguyên liệu.
2. Viết nháp từ bảng đó.
3. Rà theo tiêu chí ở mục 11.

- [ ] **Chương 1.** Nguồn: chương 1 cũ. Thêm ba thứ: mục quy đổi RMSE ra sai số
  phần trăm và ra tiền trên máy giá trung vị 18.555; bảng năm thuật toán ở mục 1.5
  theo nhóm 3 của mục 5; bản đồ chương ở mục 1.6. Đặt sợi chỉ đỏ ở cuối chương.
  Bổ sung phần giải thích cho năm công thức thuộc nhóm 1 và nhóm 2.

- [ ] **Chương 2.** Nguồn: chương 2 cũ, giữ nguyên số liệu. Đóng khung lại theo
  một trục duy nhất: mỗi quyết định làm bài toán dễ hay khó đi, đo bằng $\kappa$.
  Ba số liệu phải có: biến tương tác trên thang gốc cho $L = 17{,}14$ và 48 trị
  riêng dưới $10^{-6}$ so với trên cột đã chuẩn hóa cho $L = 2{,}68$ và 7 trị
  riêng; đường CV phẳng tới bốn chữ số thập phân từ $10^{-6}$ đến $10^{-3}$;
  quy tắc một sai số chuẩn cho $\lambda = 0{,}01$ với cái giá là RMSE 0,20601 so
  với 0,20579.

- [ ] **Chương 3.** Chỉ viết sau khi giai đoạn B xong. Nguồn: hai hình mới và
  ngưỡng $\eps_{\text{app}}$. Kết chương phải nói rõ từ đây mọi bảng báo cáo hai
  mốc.

- [ ] **Chương 4.** Nguồn: chương 6 và chương 7 cũ gộp lại. Ba hình: `all_methods_iter`,
  `all_methods_time`, `rmse_vs_time`. Bảng tổng hợp thêm cột thời gian đạt
  $\eps_{\text{app}}$. Đưa `Ridge` và `SGDRegressor` vào cùng bảng; phần quy đổi
  hàm mục tiêu xuống phụ lục C. Kết luận chính: chất lượng dự đoán như nhau, chi
  phí chênh 580 lần, ngoại lệ duy nhất là SGD bước hằng. Kết chương bằng ba câu
  hỏi còn nợ, dẫn sang chương 5 và 6.

- [ ] **Chương 5.** Nguồn: chương 3 cũ, rút gọn còn năm thuật toán. Phần kiểm thử
  của chương 4 cũ chuyển sang phụ lục A, phần ba thuật toán mở rộng sang phụ lục B.
  Bổ sung phần giải thích cho năm công thức thuộc nhóm 3.

- [ ] **Chương 6.** Nguồn: bảy mục của chương 5 cũ, số liệu giữ nguyên, sắp lại
  thành ba mục theo bảng ở mục 3.2. Đưa ba lỗi ở mục 12 vào đúng cơ chế. Bổ sung
  phần giải thích cho bốn công thức thuộc nhóm 4.

- [ ] **Chương 7.** Nguồn: mục 6.2 cũ, nâng thành chương. Hình `theory_vs_practice`.
  Số liệu phải có: cận mô tả đúng phần đuôi tới năm chữ số thập phân, nhưng GD đạt
  $10^{-6}$ sau 20 vòng lặp trong khi ước lượng từ lý thuyết là khoảng 1850. Cơ
  chế: phổ trị riêng tập trung, không trải đều giữa $\mu$ và $L$ như trường hợp
  xấu nhất mà cận nhắm tới. Bổ sung phần giải thích cho hai công thức thuộc nhóm 5.

- [ ] **Chương 8.** Nguồn: chương 8 cũ. Bảng $\lambda$, $\mu$, $\kappa$, vòng lặp
  đạt $10^{-6}$, thời gian, RMSE test. Kết luận: $\lambda$ giúp hội tụ nhanh không
  phải là $\lambda$ cho RMSE tốt nhất.

- [ ] **Chương 9.** Viết lại hoàn toàn, theo câu hỏi ở chương 1 chứ không theo
  danh sách thuật toán. Mỗi đoạn là một lời khuyên kèm điều kiện đảo chiều. Dành
  một đoạn cho câu hỏi kết luận nào là sản phẩm của dạng bậc hai và sẽ đổi trên
  hàm phi tuyến.

- [ ] **Phụ lục.** A kiểm thử, B ba thuật toán đã cắt, C quy đổi hàm mục tiêu của
  scikit-learn, D dẫn xuất công thức, E cấu hình máy và seed.

- [ ] **Bảng ký hiệu** ở đầu báo cáo, giá trị lấy từ mục 2.1.

- [ ] **Dọn `refs.bib`.** Mục nào không được trích dẫn thì xóa.

---

## 9. Giai đoạn D: viết lại slide

- [ ] Dựng lại `report/slides.tex` theo sáu phần ở mục 4, mỗi phần bám đúng chương
  tương ứng của báo cáo.
- [ ] Chuyển phần khảo sát tham số chi tiết, ba thuật toán đã cắt, phần kiểm thử
  và quy đổi hàm mục tiêu xuống phụ lục slide.
- [ ] Kiểm tra bản riêng cho slide ở `results/figures/slides/`: hình nào cần bớt
  đường hoặc phóng to chữ thì lưu vào đó, giữ nguyên tên file.
- [ ] Cập nhật `report/slides-notes.tex`.

---

## 10. Cấu trúc mã nguồn

```
ridge-phone-pricing/
├── CLAUDE.md
├── KE_HOACH_TRIEN_KHAI.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # file tải từ Kaggle, ngoài git
│   └── processed/              # feature_names.json, lambda_cv.json, problem_config.json
├── docs/
│   ├── van-phong-tieng-viet.md
│   ├── quy-uoc-bao-cao.md
│   ├── giai-thich-de-thuyet-trinh.md
│   └── superpowers/specs/
├── src/
│   ├── problem.py              # RidgeProblem: f, grad, hess, closed_form, L, mu
│   ├── history.py              # Recorder, OptimizeResult
│   ├── line_search.py          # backtracking Armijo
│   ├── first_order.py          # gradient_descent, sgd, accelerated_gradient, heavy_ball, adam
│   ├── second_order.py         # newton, newton_cg, lbfgs
│   ├── baselines.py            # so sánh với scikit-learn, hàm rmse
│   ├── runner.py               # chạy lưới tham số, run_or_load, lưu JSON
│   ├── plotting.py             # hàm vẽ chuẩn, bảng màu dùng chung
│   └── slide_figures.py        # bản hình riêng cho slide
├── notebooks/                  # 01 tới 06, chỉ gọi hàm và vẽ
├── results/
│   ├── raw/                    # kết quả dạng JSON và summary_all_methods.csv
│   └── figures/                # mỗi hình một bản PDF và một bản PNG
├── tests/
└── report/
    ├── preamble.tex
    ├── report.tex
    ├── slides.tex
    ├── slides-notes.tex
    ├── refs.bib
    └── README.md
```

Toàn bộ logic thuật toán nằm trong `src/`, notebook chỉ gọi và vẽ.

### 10.1. Giao diện chung của các thuật toán

Mọi thuật toán dùng chung chữ ký hàm và trả về cùng một cấu trúc lịch sử, nhờ vậy
mã vẽ hình dùng lại được cho mọi phương pháp.

```python
def optimizer(problem, w0=None, max_iter=1000, step_rule=None, tol=1e-10,
              record_every=1, seed=None, patience=None) -> OptimizeResult:
    ...
```

`OptimizeResult` ở `src/history.py` giữ `iter_hist`, `f_hist`, `gnorm_hist`,
`time_hist`, `access_hist`, `metric_hist`, `status`, `n_f_evals`, `n_grad_evals`,
`n_iter`, `total_time`, và `w_final`.

Hai điểm kỹ thuật quyết định tính đúng của mọi biểu đồ theo trục thời gian:

1. Thời gian ghi log không được tính vào thời gian chạy. `Recorder` dừng đồng hồ
   trước khi tính $f(w_k)$, chuẩn gradient và monitor, rồi chạy lại đồng hồ sau đó.
   Bộ đếm số lần đánh giá hàm cũng được khôi phục để việc theo dõi không làm sai
   con số báo cáo.
2. Chạy warm-up trước khi đo. `warm_up` ở `src/history.py` gọi `f_and_grad` vài
   lần để lần gọi numpy đầu tiên không bị tính chi phí khởi tạo.

---

## 11. Kiểm thử và nghiệm thu

Chạy trước mỗi lần nộp:

```bash
pytest tests/
cd report && latexmk -xelatex report.tex && latexmk -xelatex slides.tex
```

`tests/test_style.py` và `tests/test_report.py` là hai phép kiểm chạy trên nguồn
LaTeX; chúng thay cho các lượt rà bằng mắt.

Tiêu chí nghiệm thu:

- `latexmk -xelatex` biên dịch được cả `report.tex` và `slides.tex`.
- `pytest tests/test_report.py` và `pytest tests/test_style.py` xanh.
- Mọi hình và bảng có `\caption`, có `\label`, được `\ref` ít nhất một lần.
- Mục nào trong `refs.bib` không được trích dẫn thì đã xóa.
- Mỗi chương đóng bằng một câu quy ra giây, RMSE, hoặc tiền.
- Mỗi công thức trong mạch chính có đủ ba phần ở mục 5.
- Không thuật toán nào trong mạch chính thiếu mặt ở bảng tóm tắt mục 1.5.
- Bốn phép kiểm tra tính đúng đắn ở phụ lục A vẫn chạy được: gradient bằng sai
  phân hữu hạn, Hessian bằng sai phân của gradient, $\|\nabla f(w^*)\|$ ở mức sai
  số máy, và mọi thuật toán hội tụ về cùng $w^*$ trên bài toán nhỏ $n = 100$,
  $d = 5$.

---

## 12. Ghi chép quá trình thực hiện

Mục này ghi những chỗ thực tế khác với kế hoạch ban đầu, kèm lý do. Cần đọc trước
khi thuyết trình, vì câu hỏi thường rơi đúng vào đây. Từ bản kế hoạch này, ba lỗi
ở mục 12.5 không còn nằm riêng ở đây nữa mà được đưa vào chương 6 của báo cáo.

### 12.1. Quy mô bài toán

Bộ dữ liệu có 1.000.000 dòng và 28 cột, không có giá trị thiếu và không có dòng
trùng. Thí nghiệm chạy trên mẫu ngẫu nhiên 200.000 dòng, chia thành 160.000 điểm
huấn luyện và 40.000 điểm kiểm tra, $d = 280$ sau khi mã hóa.

Lý do lấy mẫu là chi phí tính toán, không phải thống kê. Với toàn bộ một triệu
dòng và $d = 280$, ma trận thiết kế chiếm 2,4 GB và một lần tính gradient mất
271 ms thay vì 39 ms, khiến toàn bộ lưới tham số mất hàng chục giờ.

### 12.2. Biến tương tác phải tạo trên cột đã chuẩn hóa

Tạo tích trên thang gốc, tức nhân `original_price` (cỡ $10^5$) với
`screen_size_inches` (cỡ 6), cho $L = 17{,}14$ và 48 trị riêng dưới $10^{-6}$.
Tạo tích trên các cột đã chuẩn hóa cho $L = 2{,}68$ và 7 trị riêng dưới $10^{-6}$,
đúng bằng mức của khối cột gốc. Không gian cột không đổi, chỉ số điều kiện tốt
lên 6,4 lần.

### 12.3. Số điều kiện do $\lambda$ quyết định, không do dữ liệu

Ma trận Gram có hạng 273 trên 280, và trị riêng kế tiếp sau 7 hướng suy biến chỉ
cỡ $2 \cdot 10^{-8}$. Do đó $\mu = \lambda$ và $\kappa = L/\lambda$ với mọi
$\lambda$ thực tế. Điều này làm chương 8 của báo cáo quan trọng hơn dự kiến.

### 12.4. Cách chọn $\lambda$ đã thay đổi

Kế hoạch dự định lấy điểm cực tiểu của đường cong cross-validation. Đường cong
thực tế phẳng hoàn toàn từ $\lambda = 10^{-6}$ đến $10^{-3}$, sai số giống nhau
tới bốn chữ số thập phân, trong khi $\kappa$ giữa hai đầu chênh nhau 1000 lần.
Lấy cực tiểu trong tình huống đó là lựa chọn tùy tiện.

Thay bằng quy tắc một sai số chuẩn: chọn $\lambda$ lớn nhất còn nằm trong một sai
số chuẩn của giá trị tốt nhất. Kết quả $\lambda = 0{,}01$, $\kappa = 268{,}3$.
Cái giá là RMSE trên tập kiểm tra 0,20601 so với 0,20579, kém đi 0,1%.

### 12.5. Ba lỗi phát hiện qua thí nghiệm

**Nhận diện cột số có đơn vị.** Quy tắc ban đầu chỉ tìm chữ số trong chuỗi, nên
`model_36` bị chuyển thành số 36 và cột định tính có nhiều mức nhất biến mất khỏi
khối one-hot. Đã siết thành quy tắc "số đứng trước, theo sau là đơn vị ngắn".

**Momentum không nhất quán với độ dài bước.** Công thức
$\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ chỉ đúng khi $t = 1/L$. Backtracking
chấp nhận bước lớn hơn nhiều khi gradient nằm theo hướng phẳng, và ghép bước đó
với momentum tính cho $1/L$ làm hàm mục tiêu bùng lên $1{,}47 \cdot 10^{4}$. Đã
viết lại thành

$$
\beta = \frac{1 - \sqrt{t\mu}}{1 + \sqrt{t\mu}}
$$

theo bước thực tế. Thí nghiệm sau đó cho thấy backtracking cho phương pháp tăng
tốc cần $\alpha = 0{,}5$, đúng bằng điều kiện chặn trên bậc hai của bổ đề giảm,
hoặc cần khởi tạo $t_0 = 1/L$; với $\alpha = 0{,}3$ và $t_0 = 1$ thì không hội tụ.

**Thiếu tiêu chí dừng khi chạm giới hạn số học.** Điều kiện dừng theo chuẩn
gradient tương đối $10^{-10}$ không đạt được, vì độ phân giải của $f$ chặn chuẩn
gradient ở khoảng $10^{-9}$. Hệ quả: một lần chạy backtracking đạt sai số
$10^{-15}$ ở vòng lặp 115 rồi chạy tiếp tới 250, và 98,7% trong 72 giây là thời
gian chết. Số lần đánh giá hàm mỗi vòng lặp bị thổi lên 17 đến 36 do line search
thất bại ở mọi bước thử khi mức giảm nằm dưới độ phân giải của $f$. Đã thêm tiêu
chí dừng theo đình trệ, áp dụng cho các phương pháp tất định, có loại trừ các lần
chạy đang phân kỳ để phép thử phân kỳ khi $t > 2/L$ vẫn hoạt động.

### 12.6. Độ dài bước của SGD phải theo kích thước lô

Hằng số Lipschitz của mất mát một mẫu là $\|x_i\|^2 + \lambda$, đo được là 3109,7
so với $L = 2{,}683$, tức lớn hơn 1159 lần. Một độ dài bước an toàn cho gradient
đầy đủ làm SGD với lô kích thước 1 phân kỳ. Mỗi kích thước lô do đó nhận độ dài
bước $1/L_B$ riêng, với

$$
L_B = L + \frac{L_{\max} - L}{B},
$$

và có một thí nghiệm tách riêng dùng chung một độ dài bước cho mọi kích thước lô
để minh họa hiện tượng phân kỳ.

### 12.7. Cơ chế chống gián đoạn

Mỗi nhóm thí nghiệm ghi ra một file JSON riêng ngay khi hoàn tất, và `run_or_load`
bỏ qua nhóm nào đã có file. Lần chạy đầu bị mất do máy ngủ giữa chừng, mất 75
phút. Sau khi có cơ chế này, gián đoạn chỉ làm mất đúng nhóm đang chạy dở.

### 12.8. Cắt ba thuật toán khỏi mạch chính

Heavy ball, Newton-CG và Adam đã cài và đã chạy, nhưng chuyển xuống phụ lục.
Heavy ball chỉ khác Nesterov ở thứ tự giữa bước ngoại suy và bước gradient nên
kết luận thu được là một câu. Newton-CG ở $d = 280$ chậm hơn Newton 25 lần. Adam
không có lợi thế trên bài toán bậc hai đã chuẩn hóa cột. Lý do cắt là trọng tâm
của bài nằm ở chỗ người làm và người nghe hiểu được, không ở số lượng thuật toán.

### 12.9. Bỏ phần mất mát Huber

Kế hoạch cũ đề xuất thêm hàm mục tiêu Huber để phần Newton và phần backtracking
có nhiều nội dung hơn. Bỏ, vì bài học thật sự về Newton trên bài toán này là bước
Newton chính là nghiệm đóng, nên Newton ở đây không phải một thuật toán lặp. Nói
thẳng điều đó rẻ hơn và rõ hơn là thêm một hàm mục tiêu mới. Chương 9 dành một
đoạn cho câu hỏi kết luận nào sẽ đổi trên hàm phi tuyến.

---

## 13. Đối chiếu với yêu cầu đề bài

| Yêu cầu | Đáp ứng tại |
| --- | --- |
| Tự lập trình gradient descent | Chương 5, `src/first_order.py` |
| Tự lập trình SGD | Chương 5, `src/first_order.py` |
| Tự lập trình accelerated gradient descent | Chương 5, `src/first_order.py` |
| Tự lập trình Newton | Chương 5, `src/second_order.py` |
| Độ dài bước cố định và backtracking cho từng thuật toán | Chương 5 và chương 6, cơ chế thứ nhất |
| Thành phần hiệu chỉnh Ridge | Chương 1 và chương 8 |
| Áp dụng thêm thuật toán khác | L-BFGS ở chương 5; Heavy ball, Newton-CG, Adam ở phụ lục B |
| Dữ liệu đủ lớn về số điểm và số thuộc tính | Chương 2, $n = 160000$, $d = 280$ |
| Mỗi thuật toán thử nhiều tham số, rút kinh nghiệm chọn tham số | Chương 6 |
| So sánh các thuật toán với setup tốt nhất | Chương 4 |
| Biểu đồ trục tung là hàm mục tiêu, hai hình theo iteration và theo thời gian | Ràng buộc chung ở đầu file, áp cho mọi hình |
| Kết luận cho từng so sánh | Tiêu chí nghiệm thu ở mục 11 |
| So sánh hàm mục tiêu và thời gian với thư viện mặc định | Chương 4, chi tiết quy đổi ở phụ lục C |
| Mọi thành viên nắm được toàn bộ nội dung | Buổi tập trình bày chéo, mỗi người trình bày phần không phải của mình |
| Ghi chép sai lệch so với kế hoạch và lý do | Mục 12 |

---

## 14. Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
| --- | --- | --- |
| Chương 4 nhắc tên thuật toán trước khi mô tả chúng | Người đọc thử phải lật về sau | Bảng năm thuật toán ở mục 1.5. Nếu vẫn hụt thì đưa chương 5 lên trước chương 3, đổi lại câu trả lời lùi về khoảng trang 30 |
| RMSE không bão hòa như dự đoán | Đường `rmse_vs_gap` dốc tới tận $10^{-8}$ | Chương 3 viết theo hướng ngược lại. Không viết chương này trước khi có số |
| Monitor bị tính vào thời gian chạy | `time_to_1e-06_s` lệch trên 20% so với bản cũ | Kiểm tra vị trí gọi monitor trong `Recorder.record`, phải nằm giữa `_pause` và `_resume` |
| Báo cáo dài thêm vì phần giải thích công thức | Vượt quá bốn trang so với bản cũ ở mạch chính | Chuyển thêm dẫn xuất xuống phụ lục D |
| Bộ quy tắc văn phong mới cũng không hiệu quả | Chương 1 viết lại vẫn khô như bản cũ | Phép thử là chương 1, không phải cảm nhận về file quy tắc. Nếu hỏng thì chẩn đoán ở giai đoạn A sai, và nguyên nhân nằm chỗ khác |
| Thầy chấm theo outline cũ | Cấu trúc chương khác hẳn bản đã nộp | Hỏi trước khi nộp lại. Mục 13 cho thấy mọi yêu cầu vẫn được đáp ứng, chỉ đổi chỗ |
