# Tái cấu trúc báo cáo và slide

Ngày: 2026-08-17

## 1. Vấn đề cần giải quyết

Giáo viên đánh giá bài làm khó theo dõi, chủ đề không có dẫn dắt, và phần thuật
toán không nói được tác động tới bài toán ứng dụng. Chẩn đoán ba nguyên nhân:

**Trật tự chương theo phân loại thuật toán.** Bản hiện tại đi theo thứ tự cài
đặt, kiểm thử, khảo sát tham số, so sánh, thư viện, hệ số hiệu chỉnh. Câu trả
lời chính nằm ở chương 6, sau khoảng bốn mươi trang khảo sát mà chưa chỗ nào nói
vì sao phải khảo sát.

**Không có đại lượng nào nối tối ưu hóa với ứng dụng.** Mọi biểu đồ dùng trục
tung $f(w_k) - f^*$. RMSE chỉ xuất hiện ở bốn chỗ lẻ và không chỗ nào nối nó với
việc chọn thuật toán, nên người đọc không trả lời được câu hỏi hiển nhiên nhất là
chọn thuật toán khác thì giá dự đoán sai lệch bao nhiêu.

**Bề rộng lấn chiều sâu.** Tám cài đặt cho bốn thuật toán bắt buộc, bảy mục khảo
sát tham số rời rạc, mười sáu công thức phần lớn được phát biểu mà không giải
thích nguồn gốc.

Số liệu chữa được hai lỗi đầu đã nằm sẵn trong `results/raw/summary_all_methods.csv`:
RMSE trên tập kiểm tra của GD, AGD, Newton và L-BFGS trùng nhau tới chữ số thứ
bảy (0,2060141), SGD bậc thang lệch ở chữ số thứ sáu, SGD bước hằng kém hẳn
(0,20986), trong khi thời gian chênh 580 lần giữa Newton (0,076 giây) và GD
(44 giây).

## 2. Quyết định đã chốt

| Quyết định | Nội dung |
| --- | --- |
| Phạm vi | Viết lại toàn bộ `report/report.tex` và `report/slides.tex` từ dàn bài mới, không vá từng chỗ |
| Thuật toán trong mạch chính | Năm: GD (bước cố định và backtracking), SGD, AGD, Newton, L-BFGS |
| Thuật toán chuyển xuống phụ lục | Heavy ball, Newton-CG, Adam. Giữ một mục ngắn nói rõ đã cài và vì sao không so sánh |
| Mất mát Huber | Không làm. Bài học về Newton trên hàm bậc hai đã đủ nếu nói thẳng; thêm hàm mục tiêu mới là thêm bộ máy chứ không thêm hiểu biết |
| Tổ chức chương giải thích | Theo ba cơ chế, không theo thuật toán |
| Chương về hệ số hiệu chỉnh | Giữ riêng, không gộp vào chương kết |
| Liều lượng toán | Mỗi công thức trong mạch chính trả lời ba câu hỏi; dẫn xuất dài xuống phụ lục; mạch chính thêm không quá bốn trang |
| Thí nghiệm mới | Đúng một: ghi RMSE trên tập kiểm tra theo từng vòng lặp |
| Bộ quy tắc văn phong | Tạm gác `docs/van-phong-tieng-viet.md`; sửa sau khi bản viết mới định hình |

Yêu cầu "áp dụng thêm thuật toán khác" của đề bài vẫn đủ nhờ L-BFGS trong mạch
chính, và ba thuật toán còn lại vẫn hiện diện ở phụ lục nên công sức đã bỏ ra
không bị giấu đi.

## 3. Nguyên tắc dẫn dắt

Sợi chỉ đỏ, đặt ở cuối chương 1 và nhắc lại ở đầu mỗi chương: *huấn luyện mô hình
định giá là giải một bài toán tối ưu hóa; giải bằng cách nào, và giải tới mức nào
thì dừng?*

Hai quy ước áp cho toàn báo cáo:

1. **Trả lời trước, giải thích sau.** Câu trả lời chính nằm ở chương 3 và 4, tức
   khoảng trang 15 thay vì trang 45. Các chương sau đọc như phần giải thích.
2. **Mỗi chương đóng bằng một câu quy ra ngôn ngữ ứng dụng**, tức bằng giây, bằng
   RMSE, hoặc bằng tiền. Đây là cơ chế giữ hai trục gắn với nhau.

## 4. Cấu trúc báo cáo

| Chương | Câu hỏi chương trả lời | Nguồn nội dung |
| --- | --- | --- |
| 1. Bài toán định giá và hàm mục tiêu | Bài toán ứng dụng là gì, đo một lời giải tốt bằng gì, cái gì quyết định độ khó | Chương 1 cũ, thêm mục quy đổi RMSE ra tiền, bảng năm thuật toán, bản đồ chương |
| 2. Chuẩn bị dữ liệu | Mỗi quyết định chuẩn bị dữ liệu làm bài toán dễ hay khó đi, đo bằng $\kappa$ | Chương 2 cũ, giữ số liệu, đóng khung lại theo một trục |
| 3. Tối ưu tới mức nào là đủ | Sai số $f - f^*$ nhỏ tới đâu thì sai số giá ngừng giảm | Chương mới, cần thí nghiệm E1 |
| 4. Thuật toán nào về đích trước | Năm thuật toán ở cấu hình tốt nhất, đặt cạnh `Ridge` và `SGDRegressor` | Gộp chương 6 và 7 cũ, thêm hình RMSE theo giây |
| 5. Cài đặt | Năm thuật toán cài thế nào, chữ ký hàm chung, cách đo thời gian | Chương 3 cũ, rút gọn |
| 6. Ba cơ chế giải thích mọi chênh lệch | Vì sao các đường nằm ở đó | Chương 5 cũ, sắp xếp lại toàn bộ |
| 7. Lý thuyết so với thực nghiệm | Cận lý thuyết đúng chỗ nào, sai chỗ nào | Mục 6.2 cũ, nâng thành chương |
| 8. Kết luận bền tới đâu | Đổi $\lambda$ tức đổi $\kappa$ thì thứ hạng có đổi không | Chương 8 cũ, đóng khung lại |
| 9. Kết luận | Trả lời câu hỏi ở chương 1, dạng lời khuyên kèm điều kiện đảo chiều | Chương 9 cũ, viết lại |

Phụ lục A: kiểm thử tính đúng đắn. Phụ lục B: ba thuật toán không đưa vào so
sánh chính. Phụ lục C: quy đổi hàm mục tiêu của scikit-learn. Phụ lục D: dẫn xuất
công thức. Phụ lục E: cấu hình máy và seed.

### 4.1. Chương 3, bản lề của báo cáo

Nội dung: RMSE trên tập kiểm tra vẽ theo $f(w_k) - f^*$, gộp mọi vòng lặp của mọi
thuật toán trên cùng một hình. Tìm ngưỡng mà RMSE ngừng cải thiện, gọi là
$\eps_{\text{app}}$, rồi quy chênh lệch RMSE ra sai số giá theo phần trăm và theo
tiền trên máy có giá trung vị 18.555.

Từ chương 4 trở đi, mọi bảng báo cáo hai mốc: thời gian đạt $\eps_{\text{app}}$
và thời gian đạt $10^{-6}$.

**Dự đoán chưa kiểm chứng:** RMSE bão hòa từ khoảng $10^{-4}$ trở xuống. Nếu thí
nghiệm E1 cho thấy RMSE tiếp tục giảm tới tận $10^{-8}$ thì kết luận của chương
đảo chiều, và chương phải viết theo hướng ngược lại, tức mọi chữ số của sai số
tối ưu hóa đều mua được độ chính xác cho bài toán định giá. Không được viết
chương này trước khi có số.

### 4.2. Chương 6, ba cơ chế

Bảy mục khảo sát tham số hiện tại giữ nguyên số liệu nhưng sắp lại thành ba mục
theo cơ chế sinh ra hiện tượng. Người nghe cần nhớ ba thứ thay vì bảy.

| Cơ chế | Hiện tượng nó giải thích |
| --- | --- |
| Độ dài bước so với ngưỡng $2/L$ | GD phân kỳ ở $t = 2.1/L$ với $f$ tăng tới $5{,}4 \cdot 10^{11}$; số vòng lặp tỉ lệ nghịch với $t$ theo dãy 20, 40, 80; chi phí backtracking do $\beta$ quyết định chứ không do $\alpha$; momentum phải khớp với bước thực tế, ghép sai làm $f$ bùng lên $1{,}47 \cdot 10^4$ |
| Nhiễu của gradient ngẫu nhiên | $L_{\max}$ lớn hơn $L$ 1159 lần nên độ dài bước phải theo kích thước lô; bước hằng dừng ở một lân cận quanh $w^*$; giảm theo bậc thang hơn bước hằng 165 lần với cùng chi phí |
| Chi phí mỗi vòng lặp | Newton cần một vòng nhưng vòng đó tốn $\bigO(nd^2 + d^3)$; L-BFGS đứng giữa hai nhóm; thứ tự các đường đổi khi trục hoành chuyển từ vòng lặp sang giây |

Ba lỗi ghi ở mục 13.5 của `KE_HOACH_TRIEN_KHAI.md` đưa vào đúng cơ chế tương ứng,
không giấu đi: chúng là bằng chứng thực nghiệm mạnh nhất cho hai cơ chế đầu.

## 5. Xử lý công thức toán học

Mỗi công thức trong mạch chính trả lời ba câu hỏi theo thứ tự: **nó nói gì** (đọc
thành lời, ưu tiên cách đọc hình học), **nó từ đâu ra** (dẫn xuất ngắn hoặc nguồn),
**nó dùng vào việc gì ở bài này** (số cụ thể và chương khai thác). Bản hiện tại
phần lớn chỉ có phần thứ ba.

Mười sáu công thức, gồm mười bốn công thức đã có nhãn cộng cập nhật của GD và đệ
quy hai vòng của L-BFGS hiện chưa đánh nhãn, chia theo vai trò chứ không theo thứ
tự xuất hiện.

**Nhóm 1, định nghĩa bài toán. Chương 1.**

| Công thức | Phần cần bổ sung |
| --- | --- |
| `eq:objective` | Vì sao có hệ số $1/2$ và vì sao chia cho $n$; công thức thể hiện đánh đổi giữa khớp dữ liệu và giữ tham số nhỏ |
| `eq:derivatives` | Hessian là ma trận hằng, không phụ thuộc $w$; đây là tính chất khiến phần còn lại của báo cáo đơn giản đi |
| `eq:closed-form` | Tồn tại và duy nhất nhờ $\lambda > 0$; hệ quả là $f^*$ biết trước nên sai số đo được là tuyệt đối |

**Nhóm 2, ba con số quyết định tốc độ. Chương 1, dùng lại ở chương 7 và 8.**

| Công thức | Phần cần bổ sung |
| --- | --- |
| `eq:constants` | Với Hessian hằng, hằng số Lipschitz của gradient chính là chuẩn phổ của Hessian. Cách đọc hình học của $\kappa$: đường mức là ellipsoid, $\kappa$ là tỉ số trục dài trên trục ngắn, $\kappa$ lớn nghĩa là thung lũng hẹp và gradient descent đi zigzag ngang thung lũng |
| `eq:mu-equals-lambda` | Ma trận Gram hạng 273 trên 280, trị riêng kế tiếp cỡ $2 \cdot 10^{-8}$; hệ quả là độ khó bài toán do người làm chọn qua $\lambda$ |

**Nhóm 3, cập nhật của từng thuật toán. Chương 5, và bảng tóm tắt đặt ở mục 1.5.**
Mỗi công thức trả lời cùng hai câu: dùng thông tin gì của hàm, trả giá bằng chi
phí gì mỗi vòng lặp.

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

Thêm hai thứ:

- **Bảng ký hiệu ở đầu báo cáo.** Một trang, liệt kê $n$, $d$, $X$, $y$, $w$,
  $\lambda$, $t$, $\eta$, $L$, $\mu$, $\kappa$, $f^*$, cột cuối ghi giá trị cụ thể
  của bài toán này.
- **Phụ lục D.** Dẫn xuất từ `eq:objective` ra `eq:derivatives` rồi ra
  `eq:closed-form`; chứng minh hằng số Lipschitz bằng chuẩn phổ của Hessian; dẫn
  `eq:minibatch-smoothness`; chứng minh `eq:gd-rate` cho hàm bậc hai.

## 6. Thí nghiệm mới

Đúng một thí nghiệm, ký hiệu E1.

**Nội dung.** Ghi RMSE trên tập kiểm tra tại từng lần ghi lịch sử, cho mọi thuật
toán ở cấu hình tốt nhất.

**Cách làm.** `HistoryRecorder.record` ở `src/history.py:187` đã nhận `w` và đã
dừng đồng hồ trước khi tính, nên thêm một callback tính RMSE trong đúng cửa sổ đã
dừng đồng hồ. Thời gian đo của thuật toán không bị ảnh hưởng.

**Sản phẩm.** Trường `rmse_hist` trong `OptimizeResult`, ghi vào JSON ở
`results/raw/`. Hai hình mới: RMSE theo $f - f^*$ cho chương 3, RMSE theo giây cho
chương 4.

**Ràng buộc.** Chạy lại các nhóm thí nghiệm đã có để sinh trường mới; giữ nguyên
seed đã ghi trong kết quả cũ để mọi con số khác không đổi.

## 7. Cấu trúc slide

Sáu phần, bám đúng mạch báo cáo:

1. Bài toán định giá và cách đo một lời giải tốt
2. Tối ưu tới mức nào là đủ, một hình
3. Thuật toán nào về đích trước, ba hình và một bảng
4. Ba cơ chế giải thích chênh lệch
5. Lý thuyết đúng ở đuôi, sai ở giai đoạn đầu
6. Kết luận dạng lời khuyên kèm điều kiện

Phụ lục slide giữ phần khảo sát tham số chi tiết, ba thuật toán đã cắt, phần kiểm
thử, và quy đổi hàm mục tiêu của thư viện.

## 8. Ánh xạ từ bản cũ sang bản mới

Bảng này để không mất nội dung khi viết lại toàn bộ.

| Chương bản cũ | Đi đâu |
| --- | --- |
| 1. Đặt vấn đề và phát biểu bài toán | Chương 1 |
| 2. Chuẩn bị dữ liệu | Chương 2 |
| 3. Các thuật toán đã cài đặt | Chương 5, phần mở rộng xuống phụ lục B |
| 4. Kiểm thử tính đúng đắn | Phụ lục A |
| 5. Khảo sát tham số, bảy mục | Chương 6, sắp lại theo ba cơ chế |
| 6. So sánh tổng hợp | Chương 4, riêng mục đối chiếu lý thuyết thành chương 7 |
| 7. So sánh với thư viện | Chương 4, phần quy đổi hàm mục tiêu xuống phụ lục C |
| 8. Ảnh hưởng của hệ số hiệu chỉnh | Chương 8 |
| 9. Kết luận | Chương 9, viết lại |

## 9. Rủi ro

**Chương 4 nhắc tên thuật toán trước khi mô tả chúng.** Đây là hệ quả của việc
trả lời sớm. Giảm nhẹ bằng bảng năm thuật toán ở mục 1.5. Nếu khi viết thấy vẫn
hụt thì phương án dự phòng là đưa chương 5 lên trước chương 3, đổi lại câu trả lời
lùi về khoảng trang 30.

**Chương 3 phụ thuộc một kết quả chưa có.** Xem mục 4.1: không viết chương này
trước khi chạy E1.

**Báo cáo dài thêm vì phần giải thích công thức.** Ước chừng bốn trang cho mạch
chính và ba tới bốn trang phụ lục. Bù bằng phần cắt ở mục 2 và bằng việc chuyển
kiểm thử xuống phụ lục.

## 10. Việc hoãn lại

**Sửa `docs/van-phong-tieng-viet.md` và mục 2 của `CLAUDE.md`.** Bộ quy tắc hiện
tại bị đánh giá là siết quá tay và làm kết quả tệ đi. Sửa sau khi bản viết mới
định hình, để lấy chính bản mới làm căn cứ thay vì sửa mò. Bộ đối chứng ở
`style/doi-chung/` giữ nguyên để đo trước sau.

**Bổ sung `style/mau/`.** Thư mục này không có trên máy, chỉ còn `style/doi-chung/`.
Quyết định giữ hay bỏ cơ chế bộ mẫu thuộc về lần sửa bộ quy tắc.

## 11. Tiêu chí nghiệm thu

- `latexmk -xelatex` biên dịch được cả `report.tex` và `slides.tex`.
- `pytest tests/test_report.py` xanh.
- Mọi hình và bảng có `\caption`, có `\label`, được `\ref` ít nhất một lần.
- Mục nào trong `refs.bib` không được trích dẫn thì xóa.
- Mỗi chương đóng bằng một câu quy ra giây, RMSE, hoặc tiền.
- Mỗi công thức trong mạch chính có đủ ba phần ở mục 5.
- Không thuật toán nào trong mạch chính thiếu mặt ở bảng tóm tắt mục 1.5.
