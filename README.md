# Tối ưu hóa hàm mất mát Ridge cho bài toán định giá điện thoại đã qua sử dụng

Bài tập môn Tối ưu hóa nâng cao, lớp Khoa học dữ liệu. Nội dung là tự cài đặt và
so sánh các thuật toán tối ưu hóa bậc một và bậc hai trên bài toán hồi quy tuyến
tính có hiệu chỉnh Ridge.

Kế hoạch chi tiết: [`KE_HOACH_TRIEN_KHAI.md`](KE_HOACH_TRIEN_KHAI.md).
Quy tắc làm việc trong thư mục: [`CLAUDE.md`](CLAUDE.md), kèm hai file chi tiết
[`docs/van-phong-tieng-viet.md`](docs/van-phong-tieng-viet.md) và
[`docs/quy-uoc-bao-cao.md`](docs/quy-uoc-bao-cao.md).

## Bài toán

$$
\min_{w \in \mathbb{R}^d} \quad f(w) = \frac{1}{2n} \left\| Xw - y \right\|_2^2 + \frac{\lambda}{2} \left\| w \right\|_2^2
$$

Hàm mục tiêu lồi mạnh và có nghiệm đóng, nên $f^*$ tính được chính xác và mọi
biểu đồ hội tụ đều vẽ $f(w_k) - f^*$ trên thang logarit.

## Cài đặt môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lấy dữ liệu

Bộ dữ liệu không nằm trong repo. Tải từ Kaggle:
<https://www.kaggle.com/datasets/sharmajicoder/used-phone-price-prediction-dataset>

Cách 1, tải thủ công: bấm Download trên trang Kaggle, giải nén, đặt file `.csv`
vào `data/raw/`.

Cách 2, dùng API (`kaggle` đã có trong `requirements.txt`): tạo API token ở
<https://www.kaggle.com/settings> mục API, tải file `kaggle.json` về, rồi

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
.venv/bin/kaggle datasets download -d sharmajicoder/used-phone-price-prediction-dataset -p data/raw --unzip
```

## Quy trình chạy

Bước 1, cố định bài toán. Chạy notebook `01` và `02`, hoặc dùng dòng lệnh:

```bash
.venv/bin/python -m src.data --csv data/raw/<tên file>.csv
```

Lệnh này xây ma trận thiết kế, chọn $\lambda$ bằng cross-validation 5 fold, tính
$L$, $\mu$, $\kappa$, $f^*$, và ghi kết quả vào `data/processed/`. Từ thời điểm
này, các file trong `data/processed/` không được sửa nữa: mọi so sánh giữa các
thuật toán chỉ có nghĩa khi chúng cùng làm việc trên một hàm mục tiêu.

Bước 2, chạy thí nghiệm. Notebook `03` (bậc một), `04` (bậc hai), `05` (so sánh
tổng hợp), `06` (so sánh với scikit-learn). Kết quả thô lưu vào `results/raw/`
dạng JSON, hình lưu vào `results/figures/` dưới cả hai định dạng PDF và PNG.

Bước 3, dựng bản hình dành riêng cho slide:

```bash
.venv/bin/python -m src.slide_figures
```

Hình trong báo cáo có tỉ lệ 6 trên 4, hợp với trang A4 dọc. Khung slide 16:9 rộng
hơn nhiều so với chiều cao, nên cùng một file phải thu nhỏ còn khoảng nửa bề rộng
slide mới vừa theo chiều dọc, và chữ trên hình không còn đọc được từ cuối phòng.
Lệnh trên vẽ lại đúng các kết quả đã lưu ở tỉ lệ rộng hơn và cỡ chữ lớn hơn, ghi
vào `results/figures/slides/`. Nó đọc file JSON trong `results/raw/`, không chạy
lại thí nghiệm nào.

Bước 4, biên dịch báo cáo và slide:

```bash
cd report
latexmk -xelatex report.tex
latexmk -xelatex slides.tex
```

Xem thêm [`report/README.md`](report/README.md).

## Kiểm thử

```bash
.venv/bin/python -m pytest tests/ -q
```

Bộ kiểm thử xác nhận gradient và Hessian khớp với sai phân hữu hạn, nghiệm đóng
là điểm dừng, mọi thuật toán cùng hội tụ về một nghiệm trên bài toán nhỏ, gradient
descent phân kỳ khi vượt ngưỡng $2/L$, việc ghi log không làm tăng số lần đánh giá
hàm, và quy đổi hệ số hiệu chỉnh giữa mã tự viết với scikit-learn là chính xác.

## Kiểm tra mã nguồn

```bash
.venv/bin/python -m pylint src tests
```

Cấu hình nằm ở `.pylintrc`. Ba nhóm quy tắc được nới so với mặc định, mỗi nhóm
kèm lý do ngay trong file: tên viết hoa cho ma trận và cho hằng số $L$, số tham
số và số biến cục bộ của các hàm thuật toán, và phần khung lặp dùng chung giữa
`src/first_order.py` với `src/second_order.py`. Ngoài ba nhóm đó, mã nguồn phải
đạt 10,00/10 trước khi nộp.

pylint không đọc được file `.ipynb`, nên phần mã trong notebook kiểm tra bằng
lệnh riêng:

```bash
.venv/bin/python tools/lint_notebooks.py
```

Lệnh này ghép các ô mã của từng notebook thành một module tạm, chạy pylint trên
đó, rồi quy các cảnh báo về đúng số thứ tự ô và số dòng trong ô. Bốn quy tắc bị
tắt vì chúng bắt vào quy ước của notebook chứ không phải lỗi; lý do của từng
quy tắc ghi trong docstring đầu file.

## Cấu trúc thư mục

```
src/
  problem.py        RidgeProblem: f, grad, hess, nghiệm đóng, L, mu, kappa
  data.py           xây ma trận thiết kế, chọn lambda, lưu bài toán cố định
  line_search.py    backtracking Armijo, các quy tắc chọn độ dài bước
  history.py        cấu trúc lịch sử dùng chung, bộ ghi có tạm dừng đồng hồ
  first_order.py    GD, SGD, AGD (Nesterov), heavy ball, Adam
  second_order.py   Newton, Newton-CG, L-BFGS
  runner.py         chạy lưới tham số, lấy trung vị thời gian, lưu và nạp JSON
  plotting.py       bảng màu cố định, cặp hình theo vòng lặp và theo thời gian
  baselines.py      các baseline scikit-learn và phần quy đổi hàm mục tiêu
  slide_figures.py  vẽ lại các hình ở tỉ lệ rộng, dùng cho slide
notebooks/          quy trình chạy, chỉ gọi hàm trong src và vẽ hình
tests/              kiểm thử tính đúng đắn
results/            kết quả JSON, hình cho báo cáo, và hình cho slide
report/             báo cáo LaTeX và slide Beamer
```

## Ghi chú về cách đo

Thời gian chạy chỉ tính phần tính toán của thuật toán. Đồng hồ được tạm dừng mỗi
khi ghi lịch sử, vì việc tính $f(w_k)$ ở mỗi vòng lặp có chi phí ngang một vòng
lặp gradient descent. Mỗi cấu hình chạy ba lần và lấy trung vị. Số liệu thời gian
phụ thuộc vào máy và phiên bản thư viện BLAS, nên khi báo cáo cần ghi kèm cấu hình
máy và nội dung `requirements.txt`.
