# Kế hoạch triển khai bài tập môn Tối ưu hóa nâng cao

**Chủ đề:** Ứng dụng các thuật toán tối ưu hóa bậc một và bậc hai trong bài toán hồi quy tuyến tính có hiệu chỉnh Ridge để định giá điện thoại đã qua sử dụng.

**Dữ liệu:** [Used Phone Price Prediction Dataset](https://www.kaggle.com/datasets/sharmajicoder/used-phone-price-prediction-dataset) (Kaggle).

**Lớp:** Khoa học dữ liệu.

---

## 1. Định vị bài toán và phạm vi công việc

Trọng tâm của bài tập là phần tối ưu hóa, không phải phần mô hình hóa hay feature engineering. Vì vậy toàn bộ kế hoạch dưới đây được tổ chức xoay quanh một câu hỏi duy nhất: cùng một hàm mục tiêu cố định, các thuật toán khác nhau và các bộ tham số khác nhau đưa giá trị hàm mục tiêu về gần cực tiểu nhanh chậm ra sao, xét theo số vòng lặp và theo thời gian chạy thực tế.

Điều này dẫn tới một nguyên tắc cần giữ xuyên suốt: **hàm mục tiêu phải được cố định trước khi bắt đầu các thí nghiệm tối ưu hóa.** Cụ thể, ma trận thiết kế $X \in \mathbb{R}^{n \times d}$, vector mục tiêu $y \in \mathbb{R}^n$ và hệ số hiệu chỉnh $\lambda$ được chốt ở giai đoạn chuẩn bị dữ liệu, sau đó không thay đổi nữa. Mọi so sánh giữa các thuật toán đều diễn ra trên cùng một hàm $f$.

Một lý do quan trọng khiến bài toán Ridge phù hợp với môn học: hàm mục tiêu lồi mạnh, khả vi vô hạn lần, và **có nghiệm đóng**. Nhờ đó nhóm tính được giá trị tối ưu $f^*$ chính xác, rồi vẽ sai số $f(w_k) - f^*$ trên thang logarit. Đây là cách trình bày cho phép nhìn thấy rõ tốc độ hội tụ tuyến tính, tốc độ hội tụ tăng tốc và hành vi bão hòa của SGD, những thứ sẽ bị che khuất nếu chỉ vẽ $f(w_k)$ trên thang tuyến tính.

### 1.1. Phát biểu bài toán tối ưu hóa

Ký hiệu $n$ là số điểm dữ liệu, $d$ là số thuộc tính sau khi mã hóa. Bài toán chính:

$$
\min_{w \in \mathbb{R}^d,\; b \in \mathbb{R}} \quad f(w, b) \;=\; \frac{1}{2n} \left\| Xw + b\mathbf{1} - y \right\|_2^2 \;+\; \frac{\lambda}{2} \left\| w \right\|_2^2
$$

Ghi chú thiết kế cần nêu trong báo cáo:

- Hệ số chặn $b$ **không** bị phạt. Đây là quy ước chuẩn, vì việc phạt $b$ khiến nghiệm phụ thuộc vào gốc tọa độ của $y$.
- Cách xử lý gọn: chuẩn hóa $X$ theo cột (trung bình $0$, độ lệch chuẩn $1$) và trừ trung bình khỏi $y$. Khi đó $b^* = 0$ và bài toán rút gọn về biến $w$ duy nhất, giúp công thức gradient, Hessian và hằng số Lipschitz sạch sẽ hơn.
- Hệ số $\frac{1}{2n}$ (thay vì $\frac{1}{2}$) giúp hằng số Lipschitz $L$ không phụ thuộc vào kích thước mẫu, thuận tiện khi so sánh với SGD.

Gradient và Hessian:

$$
\nabla f(w) = \frac{1}{n} X^{\top} (Xw - y) + \lambda w
$$

$$
\nabla^2 f(w) = \frac{1}{n} X^{\top} X + \lambda I \qquad \text{(hằng số, không phụ thuộc } w\text{)}
$$

Nghiệm đóng và giá trị tối ưu:

$$
w^* = \left( \frac{1}{n} X^{\top} X + \lambda I \right)^{-1} \left( \frac{1}{n} X^{\top} y \right), \qquad f^* = f(w^*)
$$

Các hằng số quyết định tốc độ hội tụ:

$$
L = \lambda_{\max}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\mu = \lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda, \qquad
\kappa = \frac{L}{\mu}
$$

trong đó $L$ là hằng số Lipschitz của gradient, $\mu$ là hệ số lồi mạnh, và $\kappa$ là số điều kiện. Nhóm cần tính và báo cáo cụ thể ba số này. Chúng là căn cứ để chọn độ dài bước, và là cơ sở để đối chiếu tốc độ hội tụ quan sát được với tốc độ lý thuyết.

### 1.2. Bài toán phụ (tùy chọn, khuyến khích làm)

Với hàm mục tiêu bậc hai, phương pháp Newton hội tụ sau **đúng một vòng lặp** với độ dài bước bằng $1$, và backtracking luôn chấp nhận ngay bước đầy đủ. Đây là một kết quả đúng và đáng nêu, nhưng nếu chỉ dừng ở đó thì phần Newton và phần backtracking sẽ khá nghèo nội dung.

Đề xuất bổ sung một hàm mục tiêu phi tuyến trên cùng bộ dữ liệu, dùng mất mát Huber kết hợp hiệu chỉnh Ridge:

$$
f_{\text{huber}}(w) = \frac{1}{n} \sum_{i=1}^{n} H_{\delta}\!\left( x_i^{\top} w - y_i \right) + \frac{\lambda}{2} \left\| w \right\|_2^2
$$

$$
H_{\delta}(r) =
\begin{cases}
\dfrac{r^2}{2}, & |r| \le \delta \\[2ex]
\delta \left( |r| - \dfrac{\delta}{2} \right), & |r| > \delta
\end{cases}
$$

Hàm này vẫn lồi và khả vi bậc một liên tục, Hessian tồn tại hầu khắp nơi, nhưng không còn là hàm bậc hai. Khi đó Newton cần nhiều vòng lặp, thể hiện được hội tụ bậc hai ở giai đoạn cuối, và backtracking trở nên có ý nghĩa thực sự. Mất mát Huber cũng có lý do thực tế trong bài toán này: dữ liệu giá điện thoại cũ thường có điểm ngoại lai.

Nếu quỹ thời gian hạn chế, phần này có thể lược bỏ. Bài làm vẫn đáp ứng đủ yêu cầu đề bài với riêng bài toán Ridge.

---

## 2. Giai đoạn 0: Chuẩn bị môi trường

Thư mục làm việc hiện chưa có gì và Python hệ thống chưa cài numpy, pandas, scikit-learn, matplotlib. Cần dựng môi trường ảo trước.

```bash
cd "/Users/huybq/Documents/work stuff/Optimization"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib scipy jupyter
pip freeze > requirements.txt
```

Việc ghi lại `requirements.txt` không phải hình thức. Kết quả đo thời gian chạy phụ thuộc vào phiên bản numpy và thư viện BLAS phía dưới, nên khi báo cáo số liệu thời gian, nhóm cần ghi kèm cấu hình máy và phiên bản thư viện.

Khởi tạo git để theo dõi thay đổi và chia việc:

```bash
git init
```

Về phần soạn thảo: báo cáo viết bằng LaTeX và slide làm bằng Beamer, biên dịch bằng XeLaTeX. Máy hiện tại đã có TeX Live 2025 với đầy đủ các gói cần dùng (`fontspec`, `polyglossia`, `beamer`, `booktabs`, `biblatex`, `algorithm`, `algpseudocode`, `listings`, `minted`, theme `metropolis`), và đã kiểm chứng biên dịch được văn bản tiếng Việt. Các thành viên khác cần cài TeX Live hoặc MacTeX bản đầy đủ. Quy ước chi tiết nằm ở `docs/quy-uoc-bao-cao.md`.

---

## 3. Giai đoạn 1: Chuẩn bị dữ liệu

Mục tiêu của giai đoạn này là tạo ra một cặp $(X, y)$ cố định, đủ lớn về cả số điểm lẫn số thuộc tính, rồi lưu lại để mọi thí nghiệm sau dùng chung.

### 3.1. Khảo sát dữ liệu

Tải bộ dữ liệu từ Kaggle về `data/raw/`. Bước đầu tiên là kiểm tra thực tế các cột có trong file, vì mô tả trên Kaggle không phải lúc nào cũng khớp hoàn toàn với dữ liệu.

Cần ghi nhận:

- Số dòng, số cột.
- Tên cột, kiểu dữ liệu, tỷ lệ giá trị thiếu của từng cột.
- Cột nào là biến mục tiêu (giá bán lại), phân phối của nó, có lệch phải mạnh hay không.
- Cột nào là định tính (hãng, model, tình trạng máy, màu sắc), cột nào là định lượng (RAM, dung lượng lưu trữ, dung lượng pin, kích thước màn hình, độ phân giải camera, tuổi máy, giá gốc).

### 3.2. Xử lý tối thiểu

Giữ ở mức tối thiểu, đúng như lưu ý của đề bài. Các bước cần thiết:

1. Loại các dòng thiếu biến mục tiêu.
2. Với cột định lượng thiếu giá trị: điền bằng trung vị.
3. Với cột định tính thiếu giá trị: gán thành một mức riêng tên `"unknown"`.
4. Chuyển các cột dạng chuỗi có chứa đơn vị (ví dụ `"128GB"`, `"6.1 inch"`) về số.
5. Loại các dòng có giá bằng $0$ hoặc âm.

### 3.3. Xây dựng ma trận thiết kế

Đây là bước quyết định việc bài toán có "đủ lớn về số thuộc tính" hay không.

- Mã hóa one-hot toàn bộ biến định tính. Riêng cột `model` (nếu có) thường có rất nhiều mức: gộp các mức xuất hiện dưới $10$ lần thành `"rare"`, phần còn lại giữ nguyên. Cách làm này thường tạo ra vài trăm cột, phù hợp yêu cầu.
- Bổ sung một số biến tương tác đơn giản nếu số cột còn ít, ví dụ $\text{RAM} \times \text{storage}$, $\text{tuổi\_máy}^2$. Không cần đi xa hơn, vì đây không phải trọng tâm.
- **Chuẩn hóa tất cả các cột về trung bình $0$, độ lệch chuẩn $1$.** Bước này bắt buộc và có vai trò tối ưu hóa trực tiếp: nó làm giảm số điều kiện $\kappa$ rất mạnh, và $\kappa$ chính là đại lượng chi phối tốc độ hội tụ của mọi phương pháp bậc một.
- Với biến mục tiêu: cân nhắc dùng $\log(\text{giá})$ thay vì giá, vì phân phối giá thường lệch phải. Sau đó trừ trung bình khỏi biến mục tiêu để loại hệ số chặn.

### 3.4. Chia dữ liệu và chốt $\lambda$

Chia train / test theo tỷ lệ $80/20$ với seed cố định.

Chọn $\lambda$ bằng cross-validation $5$ fold trên tập train, quét $\lambda$ trên lưới logarit từ $10^{-6}$ đến $10^{2}$. Ghi lại giá trị được chọn.

Cần phân biệt rõ hai việc, và nên nói rõ điều này khi thuyết trình:

- Chọn $\lambda$ là bài toán **học máy** (chọn mô hình). Làm một lần, ở giai đoạn này.
- Cực tiểu hóa $f$ với $\lambda$ đã cho là bài toán **tối ưu hóa**. Đây mới là nội dung chính của bài tập.

Ngoài giá trị $\lambda$ được chọn, nên giữ thêm hai giá trị $\lambda$ khác (một rất nhỏ, một lớn) để làm thí nghiệm phụ ở mục 5.6: $\lambda$ ảnh hưởng trực tiếp lên $\mu$, do đó lên $\kappa$, do đó lên tốc độ hội tụ. Đây là mối liên hệ giữa hiệu chỉnh và tối ưu hóa mà bài trình bày nên nêu.

### 3.5. Sản phẩm của giai đoạn

Lưu ra `data/processed/`:

- `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`
- `feature_names.json`
- `problem_config.json`: chứa $\lambda$, $n$, $d$, $L$, $\mu$, $\kappa$, $f^*$, $\|w^*\|$, seed

Từ thời điểm này, không ai được sửa các file trên nữa.

---

## 4. Giai đoạn 2: Cài đặt các thuật toán

### 4.1. Giao diện chung

Tất cả thuật toán nên có cùng chữ ký hàm và cùng trả về một đối tượng lịch sử thống nhất. Điều này làm cho việc vẽ biểu đồ so sánh trở nên đơn giản và tránh lỗi khi ghép kết quả.

```python
def optimizer(problem, w0, max_iter, step_rule, tol, record_every=1, seed=None):
    """
    Trả về dict:
      w_final     : nghiệm cuối
      f_hist      : list giá trị hàm mục tiêu tại các mốc ghi nhận
      gnorm_hist  : list chuẩn gradient
      time_hist   : list thời gian tích lũy (giây, chỉ tính phần tính toán)
      iter_hist   : list chỉ số vòng lặp
      nabla_hist  : list số lần truy cập dữ liệu tích lũy (để so sánh công bằng với SGD)
      status      : 'converged' | 'max_iter' | 'diverged'
    """
```

Hai điểm kỹ thuật cần làm đúng, nếu không toàn bộ biểu đồ theo trục thời gian sẽ sai:

1. **Không tính thời gian ghi log vào thời gian chạy.** Việc tính $f(w_k)$ ở mỗi vòng lặp có chi phí ngang một vòng lặp gradient descent. Cách xử lý: dùng `time.perf_counter()`, dừng đồng hồ trước khi tính và ghi log, chạy lại đồng hồ sau đó.
2. **Chạy warm-up trước khi đo.** Lần gọi numpy đầu tiên có chi phí khởi tạo. Chạy thử vài vòng lặp rồi bỏ kết quả đi trước khi bắt đầu đo thật.

### 4.2. Danh mục thuật toán bắt buộc

| Thuật toán | Biến thể độ dài bước | Ghi chú cài đặt |
|---|---|---|
| Gradient Descent | Cố định | Quét nhiều giá trị quanh $1/L$ và $2/(L+\mu)$ |
| Gradient Descent | Backtracking (Armijo) | Tham số $\alpha$, $\beta$, bước khởi tạo $t_0$ |
| SGD / mini-batch SGD | Cố định | Nhiều kích thước lô |
| SGD / mini-batch SGD | Giảm dần theo quy tắc định trước | Ít nhất $3$ quy tắc khác nhau |
| Accelerated GD (Nesterov) | Cố định | Hai công thức momentum, xem 4.3 |
| Accelerated GD (Nesterov) | Backtracking | Kèm điều kiện khởi động lại (restart) |
| Newton | Bước đầy đủ $t = 1$ | Giải hệ tuyến tính bằng Cholesky, không nghịch đảo ma trận |
| Newton | Damped, backtracking | Có ý nghĩa rõ rệt trên bài toán Huber |

### 4.3. Chi tiết từng thuật toán

**Gradient Descent.** Cập nhật

$$
w_{k+1} = w_k - t \, \nabla f(w_k)
$$

Lý thuyết cho biết phương pháp hội tụ khi $0 < t < 2/L$, và độ dài bước tối ưu cho hàm bậc hai lồi mạnh là $t = \dfrac{2}{L + \mu}$. Thí nghiệm nên bao gồm cả một giá trị $t > 2/L$ để quan sát hiện tượng phân kỳ. Đây là minh chứng trực quan cho vai trò của hằng số Lipschitz, nên đưa vào bài trình bày.

**Backtracking line search (Armijo).** Bắt đầu từ $t = t_0$, lặp $t \leftarrow \beta t$ cho tới khi

$$
f\!\left( w - t \nabla f(w) \right) \;\le\; f(w) - \alpha \, t \, \left\| \nabla f(w) \right\|_2^2
$$

Cần đếm và báo cáo **số lần đánh giá hàm mục tiêu**, không chỉ số vòng lặp. Đây là lý do vì sao backtracking có thể thắng về số vòng lặp nhưng thua về thời gian chạy, và đó chính là điểm mà đề bài yêu cầu vẽ hai biểu đồ riêng biệt.

**SGD và mini-batch SGD.** Ở mỗi bước lấy một lô ngẫu nhiên $\mathcal{B} \subset \{1, \dots, n\}$ với $|\mathcal{B}| = B$, và cập nhật theo gradient ước lượng

$$
g_k = \frac{1}{B} \sum_{i \in \mathcal{B}} \left( x_i^{\top} w_k - y_i \right) x_i + \lambda w_k,
\qquad
w_{k+1} = w_k - \eta_k \, g_k
$$

Cần lưu ý hai điều:

- Vì mỗi vòng lặp SGD rẻ hơn nhiều so với một vòng lặp GD, việc vẽ theo trục "số vòng lặp" là **không công bằng**. Cần vẽ thêm theo trục "số epoch" hoặc "số lần truy cập dữ liệu", và theo trục thời gian.
- SGD với độ dài bước hằng $\eta_k \equiv \eta$ **không hội tụ về $w^*$**, mà chỉ dao động trong một lân cận có bán kính cỡ $\mathcal{O}\!\left( \dfrac{\eta \sigma^2}{\mu} \right)$, với $\sigma^2$ là phương sai của gradient ngẫu nhiên. Trên biểu đồ thang log, hiện tượng này hiện ra thành một đường nằm ngang. Đây là kết quả cần được nêu rõ và giải thích, không phải lỗi cài đặt.

Các quy tắc chọn độ dài bước nên thử (đều là công thức tất định theo $k$, không phải line search):

$$
\eta_k = \eta_0, \qquad
\eta_k = \frac{\eta_0}{1 + \gamma k}, \qquad
\eta_k = \frac{\eta_0}{\sqrt{k+1}}, \qquad
\eta_k = \eta_0 \cdot 2^{-\lfloor k / (10 n_{\text{epoch}}) \rfloor}
$$

trong đó quy tắc cuối là giảm theo bậc thang, chia đôi sau mỗi $10$ epoch.

**Accelerated Gradient Descent (Nesterov).**

$$
\begin{aligned}
y_k &= w_k + \beta_k \left( w_k - w_{k-1} \right) \\
w_{k+1} &= y_k - t \, \nabla f(y_k)
\end{aligned}
$$

Thử hai công thức momentum:

- Khi biết $\mu$: $\beta = \dfrac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, hằng số. Đây là lựa chọn cho tốc độ lý thuyết tốt nhất với hàm lồi mạnh.
- Khi không giả định biết $\mu$: $\beta_k = \dfrac{k - 1}{k + 2}$, tăng dần. Với hàm lồi mạnh, công thức này gây ra dao động tuần hoàn thấy rõ trên biểu đồ.

Nên cài thêm **adaptive restart**: khi phát hiện

$$
\nabla f(y_k)^{\top} \left( w_{k+1} - w_k \right) > 0
$$

thì đặt lại $k = 0$. Kỹ thuật này khử được dao động nói trên và thường cho kết quả tốt nhất trong nhóm phương pháp bậc một. So sánh có restart và không restart là một thí nghiệm gọn nhưng cho kết quả rõ ràng.

**Newton.** Giải hệ

$$
\nabla^2 f(w_k) \, p_k = -\nabla f(w_k), \qquad w_{k+1} = w_k + t \, p_k
$$

- Dùng phân rã Cholesky (`scipy.linalg.cho_factor` / `cho_solve`) chứ không tính `np.linalg.inv`. Đây là điểm về độ ổn định số học và chi phí tính toán nên nêu trong báo cáo.
- Với bài toán Ridge, Hessian là hằng số, nên có thể phân rã một lần rồi dùng lại. Nhưng để so sánh trung thực với trường hợp tổng quát, nên đo cả hai phương án: phân rã lại mỗi vòng lặp, và phân rã một lần.
- Chi phí mỗi vòng lặp là $\mathcal{O}(nd^2 + d^3)$, so với $\mathcal{O}(nd)$ của gradient descent. Với $d$ cỡ vài trăm, khác biệt này thể hiện rất rõ trên biểu đồ theo trục thời gian.

Kết quả cần nêu thẳng: **với hàm mục tiêu bậc hai, Newton với $t = 1$ hội tụ sau đúng một vòng lặp, và bước Newton chính là nghiệm đóng.** Thật vậy, với $w_0 = 0$ ta có

$$
w_1 = w_0 - \left( \nabla^2 f \right)^{-1} \nabla f(w_0) = \left( \tfrac{1}{n} X^{\top} X + \lambda I \right)^{-1} \left( \tfrac{1}{n} X^{\top} y \right) = w^*
$$

Nói cách khác, Newton ở đây tương đương với việc giải trực tiếp hệ phương trình chuẩn tắc. Điều đáng so sánh không phải là số vòng lặp, mà là thời gian chạy của một vòng lặp Newton so với thời gian mà các phương pháp bậc một cần để đạt cùng độ chính xác.

### 4.4. Thuật toán mở rộng (khuyến khích, chọn $2$ đến $3$ cái)

| Thuật toán | Lý do đưa vào |
|---|---|
| Heavy-ball (Polyak momentum) | So sánh với Nesterov, cho thấy khác biệt giữa hai dạng momentum |
| Adam | Phổ biến trong thực tế, tiện đối chiếu với biểu đồ tham khảo trong đề bài |
| Coordinate Descent | Có nghiệm đóng theo từng tọa độ với hàm bậc hai, rất nhanh |
| L-BFGS (tự cài, bộ nhớ $m = 5$ hoặc $10$) | Nhóm quasi-Newton, chi phí $\mathcal{O}(md)$ mỗi vòng lặp thay vì $\mathcal{O}(d^3)$ |
| ISTA và FISTA | Cần thiết nếu nhóm muốn làm thêm Lasso |
| Newton-CG (Newton không chính xác) | Giải hệ Newton xấp xỉ bằng CG, giảm chi phí xuống dưới $\mathcal{O}(d^3)$ |

### 4.5. Phần Lasso (tùy chọn)

Nếu nhóm muốn có cả Ridge và Lasso như đề bài khuyến khích, cần lưu ý mục tiêu Lasso không khả vi tại $0$, nên không dùng được gradient descent thuần túy. Hướng làm đúng:

$$
f_{\text{lasso}}(w) = \frac{1}{2n} \left\| Xw - y \right\|_2^2 + \lambda \left\| w \right\|_1
$$

Dùng proximal gradient (ISTA) và biến thể tăng tốc (FISTA), với toán tử soft-thresholding

$$
\left[ \operatorname{prox}_{t\lambda \|\cdot\|_1}(v) \right]_j = \operatorname{sign}(v_j) \max\left( |v_j| - t\lambda,\; 0 \right)
$$

So sánh ISTA với FISTA cho ra một biểu đồ đẹp và dễ giải thích: tốc độ $\mathcal{O}(1/k)$ so với $\mathcal{O}(1/k^2)$. Ngoài ra có thể báo cáo thêm số hệ số bằng $0$ trong nghiệm, đối chiếu với Ridge.

---

## 5. Giai đoạn 3: Thiết kế thí nghiệm

### 5.1. Cấu hình chung cho mọi thí nghiệm

- Điểm khởi tạo: $w_0 = 0$ cho tất cả thuật toán. Với các phương pháp ngẫu nhiên, chạy $5$ seed khác nhau và báo cáo trung vị kèm dải min-max.
- Số vòng lặp tối đa: đặt theo từng nhóm phương pháp (GD và AGD: $5000$; SGD: $200$ epoch; Newton: $50$).
- Điều kiện dừng: $\left\| \nabla f(w_k) \right\| \le 10^{-10} \left\| \nabla f(w_0) \right\|$ hoặc chạm giới hạn vòng lặp. Với thí nghiệm vẽ biểu đồ, nên để chạy đủ số vòng lặp để đường cong đầy đủ.
- Chỉ số theo dõi chính: $f(w_k) - f^*$ trên thang $\log_{10}$.
- Đo thời gian: `time.perf_counter()`, mỗi cấu hình chạy $3$ lần độc lập, lấy trung vị để giảm nhiễu.

### 5.2. Lưới tham số cần quét

**GD với bước cố định.**

$$
t \in \left\{ \frac{2.1}{L},\; \frac{2}{L},\; \frac{1.9}{L},\; \frac{2}{L+\mu},\; \frac{1}{L},\; \frac{0.5}{L},\; \frac{0.1}{L},\; \frac{0.01}{L} \right\}
$$

Giá trị $2.1/L$ được đưa vào có chủ ý, để quan sát phân kỳ.

**GD với backtracking.** $\alpha \in \{0.1,\, 0.3,\, 0.5\}$, $\beta \in \{0.5,\, 0.8,\, 0.9\}$, $t_0 \in \{1,\; 10/L\}$. Với mỗi cấu hình, ghi lại số lần đánh giá hàm trung bình trên mỗi vòng lặp.

**SGD.** Kích thước lô $B \in \{1,\, 16,\, 64,\, 256,\, 1024\}$, kết hợp với các quy tắc chọn độ dài bước ở mục 4.3, $\eta_0$ quét trên lưới logarit. Đây là phần có nhiều tổ hợp nhất, nên chia việc cho hai người.

**AGD.** Momentum theo hai công thức ở mục 4.3, có và không có restart, độ dài bước $t \in \{1/L,\; 0.5/L\}$, cộng thêm biến thể backtracking.

**Newton.** $t = 1$ so với damped có backtracking; phân rã Cholesky lại mỗi vòng so với dùng lại; trên cả hai hàm mục tiêu (Ridge và Huber nếu có làm).

### 5.3. Bộ biểu đồ cần vẽ

Đề bài yêu cầu rõ: mỗi so sánh phải có hai hình, một theo số vòng lặp và một theo thời gian, trục tung là độ lớn hàm mục tiêu. Danh sách cụ thể:

| Nhóm | Nội dung so sánh | Số hình |
|---|---|---|
| A | GD, các độ dài bước cố định khác nhau | 2 |
| B | GD, các cấu hình backtracking khác nhau | 2 |
| C | SGD, các kích thước lô khác nhau | 2 |
| D | SGD, các quy tắc chọn độ dài bước khác nhau | 2 |
| E | AGD, các cấu hình momentum, có và không restart | 2 |
| F | Newton, bước đầy đủ và damped, hai cách xử lý Hessian | 2 |
| G | **So sánh tổng hợp:** mỗi thuật toán dùng cấu hình tốt nhất tìm được ở A đến F | 2 |
| H | So sánh mã tự viết với thư viện (mục 6) | 2 |
| I | Ảnh hưởng của $\lambda$ lên tốc độ hội tụ (thí nghiệm mục 5.6) | 2 |
| J | Ảnh hưởng của việc chuẩn hóa dữ liệu (thí nghiệm mục 5.7) | 2 |

Quy ước trình bày để các hình đọc được nhất quán:

- Trục tung: $\log_{10}\!\left( f(w_k) - f^* \right)$, dùng `plt.semilogy`. Nếu vì lý do nào đó không có $f^*$, dùng $f(w_k)$ nhưng sẽ khó nhìn hơn nhiều.
- Trục hoành hình thứ nhất: số vòng lặp $k$. Với SGD, dùng thêm một hình phụ theo số epoch để so sánh công bằng.
- Trục hoành hình thứ hai: thời gian tích lũy tính bằng giây.
- Mỗi thuật toán một màu cố định, dùng chung bảng màu cho toàn bộ báo cáo.
- Chú thích rõ tham số trong legend, ví dụ `GD (t = 1/L)`, không ghi chung chung là `GD`.
- Mỗi hình đi kèm ít nhất hai câu kết luận trong văn bản. Hình không có kết luận thì không tính là đã hoàn thành.

Nhóm G là hình quan trọng nhất của bài trình bày, tương ứng với hình tham khảo mà đề bài dẫn ra.

### 5.4. Bảng tổng hợp kết quả

Ngoài biểu đồ, cần một bảng số liệu để so sánh định lượng:

| Thuật toán | Cấu hình tốt nhất | Số vòng lặp đạt $f - f^* < 10^{-6}$ | Thời gian đạt ngưỡng đó (giây) | $f$ cuối cùng | RMSE trên test |
|---|---|---|---|---|---|

Cột "số vòng lặp đạt ngưỡng" hữu ích hơn cột "số vòng lặp chạy hết", vì nó là con số trực tiếp trả lời câu hỏi thuật toán nào nhanh hơn.

### 5.5. Đối chiếu với lý thuyết

Với hàm bậc hai lồi mạnh, lý thuyết cho các cận:

- GD với độ dài bước tối ưu $t = \dfrac{2}{L+\mu}$:

$$
f(w_k) - f^* \;\le\; \left( \frac{\kappa - 1}{\kappa + 1} \right)^{2k} \left( f(w_0) - f^* \right)
$$

- AGD với momentum tối ưu:

$$
f(w_k) - f^* \;\le\; \left( 1 - \frac{1}{\sqrt{\kappa}} \right)^{k} \cdot C \left( f(w_0) - f^* \right)
$$

Từ dữ liệu thực nghiệm, ước lượng hệ số co rút quan sát được bằng cách khớp đường thẳng vào phần tuyến tính của đồ thị $\log_{10}\!\left( f(w_k) - f^* \right)$ theo $k$, rồi đối chiếu với cận lý thuyết. Đây là nội dung thể hiện rõ "trải nghiệm học được" mà đề bài yêu cầu, và thường được đánh giá cao hơn việc chỉ trưng ra biểu đồ.

### 5.6. Thí nghiệm về ảnh hưởng của $\lambda$

Chạy lại GD và AGD với ba giá trị $\lambda$ khác nhau. Vì

$$
\mu = \lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda,
\qquad
\kappa = \frac{\lambda_{\max}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda}{\lambda_{\min}\!\left( \tfrac{1}{n} X^{\top} X \right) + \lambda}
$$

tăng $\lambda$ làm tăng $\mu$, giảm $\kappa$, do đó tăng tốc độ hội tụ. Thí nghiệm này cho thấy hiệu chỉnh Ridge có hai vai trò tách biệt: một vai trò thống kê (chống quá khớp) và một vai trò tối ưu hóa (cải thiện điều kiện của bài toán). Đây là điểm nối trực tiếp giữa chủ đề Ridge và nội dung môn học.

Nên báo cáo bảng: $\lambda$, $\mu$, $\kappa$, số vòng lặp GD cần để đạt $10^{-6}$, RMSE trên test. Bảng này thường cho thấy $\lambda$ giúp hội tụ nhanh không phải là $\lambda$ cho RMSE tốt nhất, một sự đánh đổi đáng bàn.

### 5.7. Thí nghiệm về ảnh hưởng của chuẩn hóa

Chạy GD trên $X$ chưa chuẩn hóa và $X$ đã chuẩn hóa, với $\kappa$ được báo cáo cho cả hai trường hợp. Chênh lệch thường rất lớn, và đây là bài học thực hành trực tiếp: chuẩn hóa dữ liệu không chỉ là thói quen tiền xử lý, mà là một can thiệp lên số điều kiện của bài toán tối ưu hóa.

---

## 6. Giai đoạn 4: So sánh với thư viện

Đề bài yêu cầu rõ: so sánh **giá trị hàm mục tiêu** và **thời gian tính toán**, không phải độ chính xác. Độ chính xác có thể báo cáo thêm.

Điểm cần cẩn thận nhất ở giai đoạn này là **quy đổi hàm mục tiêu về cùng một dạng**. Mỗi thư viện dùng một cách chuẩn hóa hệ số khác nhau:

- `sklearn.linear_model.Ridge` cực tiểu hóa

$$
\left\| Xw - y \right\|_2^2 + \alpha \left\| w \right\|_2^2
$$

  tức là không có hệ số $\frac{1}{n}$ và không có hệ số $\frac{1}{2}$.

- `sklearn.linear_model.SGDRegressor` với `penalty='l2'` cực tiểu hóa

$$
\frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\!\left( x_i^{\top} w, y_i \right) + \frac{\alpha}{2} \left\| w \right\|_2^2
$$

  tùy phiên bản.

Vì vậy nhóm phải quy đổi $\alpha$ của sklearn sang $\lambda$ của mình sao cho **hai nghiệm tối ưu trùng nhau**, rồi mới so sánh. Cách kiểm tra: lấy $\hat{w}$ do sklearn trả về, cắm vào hàm $f$ của nhóm, và so với $f^*$ đã tính từ nghiệm đóng. Nếu quy đổi đúng, chênh lệch phải ở mức sai số máy.

Với `Ridge`, quan hệ là

$$
\alpha_{\text{sklearn}} = \lambda \cdot n
$$

khi hàm mục tiêu của nhóm có hệ số $\frac{1}{2n}$ và $\frac{\lambda}{2}$. Cần kiểm chứng bằng số chứ không tin vào suy luận trên giấy.

Các đối tượng so sánh:

| Thư viện | Cấu hình | Ghi chú |
|---|---|---|
| `Ridge(solver='auto')` | Mặc định | Thường dùng Cholesky, tương đương nghiệm đóng |
| `Ridge(solver='sag')` | Mặc định | Phương pháp gradient ngẫu nhiên trung bình |
| `Ridge(solver='lsqr')` | Mặc định | Phương pháp lặp dựa trên bình phương tối thiểu |
| `SGDRegressor` | Mặc định | Đối chiếu trực tiếp với SGD tự cài |
| `LinearRegression` | Mặc định | Trường hợp $\lambda = 0$, để tham chiếu |

Bảng kết quả cần có: giá trị $f$ đạt được, sai số $f - f^*$, thời gian huấn luyện, và RMSE trên test.

Dự đoán kết quả để nhóm không bất ngờ khi trình bày: `Ridge` với solver mặc định sẽ đạt $f^*$ với sai số cỡ máy và rất nhanh, vì nó giải trực tiếp. `SGDRegressor` với tham số mặc định thường **không** hội tụ tốt trên bài toán chưa được điều chỉnh tham số, và có thể cho $f$ cao hơn đáng kể so với mã tự viết đã được tinh chỉnh. Cả hai quan sát này đều là kết quả hợp lệ và đáng bàn: chúng cho thấy tham số mặc định của thư viện không phải lúc nào cũng phù hợp, và với bài toán có nghiệm đóng thì phương pháp trực tiếp khó bị đánh bại.

Kết luận cần đưa ra một cách cân bằng: mục đích của việc tự cài đặt không phải là để chạy nhanh hơn thư viện, mà là để hiểu cơ chế hội tụ và biết cách chọn tham số. Khi bài toán lớn tới mức không giải trực tiếp được ($d$ rất lớn, chi phí $\mathcal{O}(d^3)$ không chấp nhận được), các phương pháp lặp mới thể hiện lợi thế.

---

## 7. Cấu trúc mã nguồn đề xuất

```
Optimization/
├── README.md
├── KE_HOACH_TRIEN_KHAI.md
├── requirements.txt
├── data/
│   ├── raw/                    # file tải từ Kaggle
│   └── processed/              # X_train.npy, y_train.npy, ... , problem_config.json
├── src/
│   ├── problem.py              # lớp RidgeProblem: f, grad, hess, closed_form, L, mu
│   ├── problem_huber.py        # (tùy chọn) bài toán phụ
│   ├── line_search.py          # backtracking Armijo, Wolfe
│   ├── first_order.py          # gd, sgd, agd, heavy_ball, adam
│   ├── second_order.py         # newton, newton_cg, lbfgs
│   ├── proximal.py             # (tùy chọn) ista, fista cho Lasso
│   ├── runner.py               # chạy lưới tham số, lưu kết quả ra JSON
│   └── plotting.py             # hàm vẽ chuẩn, dùng chung bảng màu
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_experiments_first_order.ipynb
│   ├── 04_experiments_second_order.ipynb
│   ├── 05_comparison_all.ipynb
│   └── 06_comparison_sklearn.ipynb
├── results/
│   ├── raw/                    # kết quả chạy dạng JSON
│   └── figures/                # hình xuất ra, mỗi hình một bản PDF và một bản PNG
└── report/
    ├── preamble.tex            # khai báo dùng chung cho báo cáo và slide
    ├── report.tex              # báo cáo, biên dịch bằng XeLaTeX
    ├── slides.tex              # slide Beamer, biên dịch bằng XeLaTeX
    ├── refs.bib                # tài liệu tham khảo
    ├── figures/                # hình PDF dùng trong báo cáo và slide
    └── README.md               # hướng dẫn biên dịch, gói LaTeX cần cài
```

Ghi chú: toàn bộ logic thuật toán nằm trong `src/`, notebook chỉ gọi và vẽ. Cách tổ chức này giúp tránh tình trạng cùng một thuật toán được sửa ở ba notebook khác nhau và cho kết quả khác nhau.

---

## 8. Kiểm thử tính đúng đắn

Trước khi chạy thí nghiệm quy mô lớn, cần kiểm tra mã. Bốn phép kiểm tra sau là tối thiểu và tốn ít công:

1. **Kiểm tra gradient bằng sai phân hữu hạn.** So sánh $\left[ \nabla f(w) \right]_i$ giải tích với sai phân trung tâm

$$
\frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}, \qquad \varepsilon = 10^{-6}
$$

   tại vài tọa độ $i$ ngẫu nhiên. Sai số tương đối phải dưới $10^{-6}$.

2. **Kiểm tra Hessian** tương tự, dùng sai phân của gradient:

$$
\left[ \nabla^2 f(w) \right]_{:,i} \approx \frac{\nabla f(w + \varepsilon e_i) - \nabla f(w - \varepsilon e_i)}{2\varepsilon}
$$

3. **Kiểm tra nghiệm đóng.** Xác nhận $\left\| \nabla f(w^*) \right\|$ ở mức sai số máy.
4. **Kiểm tra trên bài toán nhỏ có nghiệm biết trước.** Sinh dữ liệu giả với $n = 100$, $d = 5$, chạy mọi thuật toán, xác nhận tất cả đều hội tụ về cùng một $w^*$.

Nếu bỏ qua bước này, rủi ro lớn nhất là chạy hết toàn bộ thí nghiệm rồi mới phát hiện gradient sai dấu ở một nhánh nào đó.

---

## 9. Lịch trình và phân công

Giả định nhóm $4$ người. Nếu số thành viên khác, gộp hoặc tách vai trò tương ứng.

| Tuần | Nội dung | Sản phẩm |
|---|---|---|
| 1 | Dựng môi trường, khảo sát dữ liệu, chuẩn bị $X$, $y$, chốt $\lambda$, tính $L$, $\mu$, $\kappa$, $f^*$ | `data/processed/`, notebook 01 và 02 |
| 2 | Cài `problem.py`, `line_search.py`, `first_order.py`, `second_order.py`, chạy toàn bộ kiểm thử ở mục 8 | `src/` chạy được, test pass |
| 3 | Chạy lưới tham số, sinh biểu đồ nhóm A đến F, ghi kết luận cho từng nhóm | `results/`, notebook 03 và 04 |
| 4 | So sánh tổng hợp (nhóm G), so sánh sklearn (nhóm H), thí nghiệm $\lambda$ và chuẩn hóa (I, J) | notebook 05 và 06 |
| 5 | Viết báo cáo bằng LaTeX, làm slide Beamer, tập trình bày chéo | `report/report.pdf`, `report/slides.pdf` |

Phân công theo vai trò, không theo mảng kín. Mỗi người phụ trách chính một phần nhưng phải đọc và hiểu phần của người khác:

| Vai trò | Phụ trách chính | Phần phải nắm được để trình bày |
|---|---|---|
| A | Dữ liệu, `problem.py`, tính $L$, $\mu$, $\kappa$, $f^*$, thí nghiệm mục 5.6 và 5.7 | Toàn bộ |
| B | GD và backtracking, nhóm biểu đồ A và B | Toàn bộ |
| C | SGD và các quy tắc chọn bước, nhóm biểu đồ C và D | Toàn bộ |
| D | AGD và Newton, nhóm biểu đồ E và F | Toàn bộ |
| Cả nhóm | Nhóm biểu đồ G, H, phần so sánh lý thuyết, báo cáo, slide | Toàn bộ |

Đề bài nêu rõ mọi thành viên phải trình bày được bất cứ nội dung nào. Đề xuất thực hiện: trước buổi thuyết trình, tổ chức một buổi tập trong đó mỗi người trình bày phần **không phải** của mình. Cách này phát hiện được lỗ hổng hiểu biết sớm hơn nhiều so với việc mỗi người chỉ ôn phần mình.

---

## 10. Outline bài trình bày

Thời lượng dự kiến $20$ phút, khoảng $20$ slide. Phân bổ thời gian nghiêng hẳn về phần tối ưu hóa.

**Phần 1. Đặt vấn đề (2 slide, 2 phút)**

1. Bài toán định giá điện thoại đã qua sử dụng, mô tả dữ liệu ngắn gọn: $n$, $d$, vài thống kê cơ bản.
2. Phát biểu bài toán tối ưu hóa: hàm mục tiêu Ridge, gradient, Hessian, nghiệm đóng.

**Phần 2. Đặc trưng của bài toán (2 slide, 3 phút)**

3. Các hằng số $L$, $\mu$, $\kappa$ tính được, và ý nghĩa của chúng đối với tốc độ hội tụ dự kiến.
4. Ảnh hưởng của chuẩn hóa dữ liệu lên $\kappa$ (biểu đồ nhóm J). Đây là slide cho thấy tiền xử lý và tối ưu hóa không tách rời.

**Phần 3. Từng thuật toán và việc chọn tham số (8 slide, 8 phút)**

5. GD với bước cố định: biểu đồ nhóm A, gồm cả trường hợp phân kỳ khi $t > 2/L$.
6. GD với backtracking: biểu đồ nhóm B, kèm số lần đánh giá hàm mỗi vòng lặp.
7. Kết luận về chọn bước cho GD.
8. SGD, ảnh hưởng của kích thước lô $B$: biểu đồ nhóm C.
9. SGD, ảnh hưởng của quy tắc chọn bước $\eta_k$: biểu đồ nhóm D, giải thích hiện tượng bão hòa khi bước hằng.
10. AGD: biểu đồ nhóm E, so sánh hai công thức momentum, tác dụng của restart.
11. Newton: biểu đồ nhóm F, giải thích vì sao hội tụ sau một vòng lặp trên hàm bậc hai, và chi phí $\mathcal{O}(d^3)$ mỗi vòng.
12. (Nếu có) Newton trên bài toán Huber: hội tụ bậc hai ở giai đoạn cuối.

**Phần 4. So sánh tổng hợp (3 slide, 4 phút)**

13. Biểu đồ nhóm G theo số vòng lặp.
14. Biểu đồ nhóm G theo thời gian chạy. Nhấn vào chỗ thứ hạng thay đổi so với slide trước.
15. Bảng tổng hợp ở mục 5.4, và kết luận về cấu hình nên chọn.

**Phần 5. So sánh với thư viện (2 slide, 2 phút)**

16. Cách quy đổi hàm mục tiêu giữa mã tự viết và sklearn, kèm bằng chứng kiểm chứng bằng số.
17. Bảng và biểu đồ nhóm H, kết luận cân bằng như đã nêu ở mục 6.

**Phần 6. Kết luận (2 slide, 2 phút)**

18. Ảnh hưởng của $\lambda$ lên cả chất lượng dự báo lẫn tốc độ hội tụ (biểu đồ nhóm I).
19. Những điều nhóm rút ra: cách chọn độ dài bước, khi nào nên dùng backtracking, khi nào phương pháp bậc hai đáng chi phí, đối chiếu lý thuyết và thực nghiệm.

---

## 11. Đối chiếu với yêu cầu đề bài

| Yêu cầu | Đáp ứng tại |
|---|---|
| Tự lập trình gradient descent | Mục 4.3, `src/first_order.py` |
| Tự lập trình SGD | Mục 4.3, `src/first_order.py` |
| Tự lập trình accelerated gradient descent | Mục 4.3, `src/first_order.py` |
| Tự lập trình Newton | Mục 4.3, `src/second_order.py` |
| Độ dài bước cố định và backtracking cho từng thuật toán | Bảng 4.2, lưới tham số 5.2 |
| Thành phần hiệu chỉnh Ridge | Hàm mục tiêu mục 1.1, thí nghiệm 5.6 |
| Áp dụng thêm thuật toán khác (khuyến khích) | Mục 4.4 và 4.5 |
| Dữ liệu đủ lớn về số điểm và số thuộc tính | Mục 3.3, one-hot tạo vài trăm cột |
| Mỗi thuật toán thử nhiều tham số, rút kinh nghiệm chọn tham số | Mục 5.2, slide 5 đến 12 |
| So sánh các thuật toán với setup tốt nhất | Nhóm biểu đồ G, mục 5.3 |
| Biểu đồ trục tung là hàm mục tiêu, hai hình theo iteration và theo thời gian | Quy ước mục 5.3, áp dụng cho mọi nhóm biểu đồ |
| Kết luận cho từng so sánh | Yêu cầu tối thiểu hai câu kết luận mỗi hình, mục 5.3 |
| So sánh hàm mục tiêu và thời gian với thư viện mặc định | Mục 6, nhóm biểu đồ H |
| Mọi thành viên nắm được toàn bộ nội dung | Mục 9, buổi tập trình bày chéo |
| Ghi chép sai lệch so với kế hoạch và lý do | Mục 13 |

---

## 12. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Dữ liệu nhỏ hơn dự kiến, không đủ thuộc tính | $d < 30$ sau khi mã hóa | Giữ cột `model` với nhiều mức hơn, thêm biến tương tác bậc hai, hoặc bổ sung thêm một bộ dữ liệu tương tự |
| Số điều kiện $\kappa$ quá lớn, GD gần như không nhúc nhích | Đường cong GD gần nằm ngang | Đây là kết quả hợp lệ và đáng báo cáo. Tăng $\lambda$, hoặc loại các cột gần cộng tuyến để có thêm một kịch bản đối chiếu |
| Newton hội tụ một bước, phần bậc hai quá mỏng | Chỉ có một điểm trên đồ thị | Bổ sung bài toán Huber ở mục 1.2, và so sánh chi phí thời gian mỗi vòng lặp |
| Kết quả đo thời gian dao động mạnh giữa các lần chạy | Chênh lệch trên $20\%$ | Đóng ứng dụng khác, chạy $3$ đến $5$ lần lấy trung vị, đảm bảo đã loại thời gian ghi log |
| Quy đổi hàm mục tiêu với sklearn bị sai | $f(\hat{w}_{\text{sklearn}})$ khác $f^*$ đáng kể | Kiểm chứng bằng số theo cách mô tả ở mục 6, không tin suy luận trên giấy |
| Hàm mục tiêu bị thay đổi giữa chừng | Các biểu đồ không so sánh được với nhau | Chốt `data/processed/` và `problem_config.json` từ tuần 1, không sửa về sau |

---

## 13. Ghi chép quá trình thực hiện

Mục này ghi lại những chỗ thực tế khác với kế hoạch ban đầu, kèm lý do. Đây là phần cần đọc trước khi thuyết trình, vì các câu hỏi thường rơi đúng vào những chỗ này.

### 13.1. Quy mô bài toán

Bộ dữ liệu có 1.000.000 dòng và 28 cột, không có giá trị thiếu và không có dòng trùng. Thí nghiệm chạy trên mẫu ngẫu nhiên 200.000 dòng, chia thành 160.000 điểm huấn luyện và 40.000 điểm kiểm tra, $d = 280$ sau khi mã hóa.

Lý do lấy mẫu là chi phí tính toán, không phải thống kê. Với toàn bộ một triệu dòng và $d = 280$, ma trận thiết kế chiếm 2,4 GB và một lần tính gradient mất 271 ms thay vì 39 ms, khiến toàn bộ lưới tham số mất hàng chục giờ.

### 13.2. Biến tương tác phải tạo trên cột đã chuẩn hóa

Tạo tích trên thang gốc, tức nhân `original_price` (cỡ $10^5$) với `screen_size_inches` (cỡ 6), cho $L = 17{,}14$ và 48 trị riêng dưới $10^{-6}$. Tạo tích trên các cột đã chuẩn hóa cho $L = 2{,}68$ và 7 trị riêng dưới $10^{-6}$, đúng bằng mức của khối cột gốc. Không gian cột không đổi, chỉ số điều kiện tốt lên 6,4 lần.

### 13.3. Số điều kiện do $\lambda$ quyết định, không do dữ liệu

Ma trận Gram có hạng 273 trên 280, và trị riêng kế tiếp sau 7 hướng suy biến chỉ cỡ $2 \cdot 10^{-8}$. Do đó $\mu = \lambda$ và $\kappa = L/\lambda$ với mọi $\lambda$ thực tế. Điều này làm mục 5.6 quan trọng hơn dự kiến trong kế hoạch.

### 13.4. Cách chọn $\lambda$ đã thay đổi

Kế hoạch dự định lấy điểm cực tiểu của đường cong cross-validation. Đường cong thực tế phẳng hoàn toàn từ $\lambda = 10^{-6}$ đến $10^{-3}$, sai số giống nhau tới bốn chữ số thập phân, trong khi $\kappa$ giữa hai đầu chênh nhau 1000 lần. Lấy cực tiểu trong tình huống đó là lựa chọn tùy tiện.

Thay bằng quy tắc một sai số chuẩn: chọn $\lambda$ lớn nhất còn nằm trong một sai số chuẩn của giá trị tốt nhất. Kết quả $\lambda = 0{,}01$, $\kappa = 268{,}3$. Cái giá là RMSE trên tập kiểm tra 0,20601 so với 0,20579, kém đi 0,1%.

### 13.5. Ba lỗi phát hiện qua thí nghiệm

**Nhận diện cột số có đơn vị.** Quy tắc ban đầu chỉ tìm chữ số trong chuỗi, nên `model_36` bị chuyển thành số 36 và cột định tính có nhiều mức nhất biến mất khỏi khối one-hot. Đã siết thành quy tắc "số đứng trước, theo sau là đơn vị ngắn".

**Momentum không nhất quán với độ dài bước.** Công thức $\beta = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ chỉ đúng khi $t = 1/L$. Backtracking chấp nhận bước lớn hơn nhiều khi gradient nằm theo hướng phẳng, và ghép bước đó với momentum tính cho $1/L$ làm hàm mục tiêu bùng lên $1{,}47 \cdot 10^{4}$. Đã viết lại thành

$$
\beta = \frac{1 - \sqrt{t\mu}}{1 + \sqrt{t\mu}}
$$

theo bước thực tế. Thí nghiệm sau đó cho thấy backtracking cho phương pháp tăng tốc cần $\alpha = 0{,}5$, đúng bằng điều kiện chặn trên bậc hai của bổ đề giảm, hoặc cần khởi tạo $t_0 = 1/L$; với $\alpha = 0{,}3$ và $t_0 = 1$ thì không hội tụ.

**Thiếu tiêu chí dừng khi chạm giới hạn số học.** Điều kiện dừng theo chuẩn gradient tương đối $10^{-10}$ không đạt được, vì độ phân giải của $f$ chặn chuẩn gradient ở khoảng $10^{-9}$. Hệ quả: một lần chạy backtracking đạt sai số $10^{-15}$ ở vòng lặp 115 rồi chạy tiếp tới 250, và 98,7% trong 72 giây là thời gian chết. Số lần đánh giá hàm mỗi vòng lặp bị thổi lên 17 đến 36 do line search thất bại ở mọi bước thử khi mức giảm nằm dưới độ phân giải của $f$. Đã thêm tiêu chí dừng theo đình trệ, áp dụng cho các phương pháp tất định, có loại trừ các lần chạy đang phân kỳ để phép thử phân kỳ khi $t > 2/L$ vẫn hoạt động.

### 13.6. Độ dài bước của SGD phải theo kích thước lô

Hằng số Lipschitz của mất mát một mẫu là $\|x_i\|^2 + \lambda$, đo được là 3109,7 so với $L = 2{,}683$, tức lớn hơn 1159 lần. Một độ dài bước an toàn cho gradient đầy đủ làm SGD với lô kích thước 1 phân kỳ. Mỗi kích thước lô do đó nhận độ dài bước $1/L_B$ riêng, với

$$
L_B = L + \frac{L_{\max} - L}{B},
$$

và có một thí nghiệm tách riêng dùng chung một độ dài bước cho mọi kích thước lô để minh họa hiện tượng phân kỳ.

### 13.7. Cơ chế chống gián đoạn

Mỗi nhóm thí nghiệm ghi ra một file JSON riêng ngay khi hoàn tất, và `run_or_load` bỏ qua nhóm nào đã có file. Lần chạy đầu bị mất do máy ngủ giữa chừng, mất 75 phút. Sau khi có cơ chế này, gián đoạn chỉ làm mất đúng nhóm đang chạy dở.

---

## 14. Việc cần làm ngay

1. Tải bộ dữ liệu từ Kaggle về `data/raw/`.
2. Dựng môi trường ảo theo mục 2.
3. Chạy notebook khảo sát để xác nhận số dòng, số cột, và số thuộc tính $d$ dự kiến sau khi mã hóa. Nếu con số này quá nhỏ, cần điều chỉnh phương án ở mục 3.3 trước khi làm tiếp.
4. Cài `src/problem.py` và tính $L$, $\mu$, $\kappa$, $f^*$. Ba con số đầu là đầu vào cho toàn bộ phần chọn tham số phía sau, nên cần có sớm.
