# Quy tắc làm việc trong thư mục này

Dự án: bài tập môn Tối ưu hóa nâng cao, chủ đề các thuật toán tối ưu hóa bậc một và bậc hai cho bài toán hồi quy tuyến tính có hiệu chỉnh Ridge. Kế hoạch chi tiết nằm ở `KE_HOACH_TRIEN_KHAI.md`.

## 1. Ngôn ngữ

Có hai lớp nội dung, dùng hai ngôn ngữ khác nhau, không trộn lẫn.

**Tiếng Việt** dùng cho toàn bộ nội dung dành cho người đọc: file markdown, báo cáo, slide, phần văn bản trong notebook (ô markdown), tiêu đề và chú thích trong bài trình bày.

**Tiếng Anh** dùng cho toàn bộ mã nguồn, không có ngoại lệ:

- Tên biến, tên hàm, tên lớp, tên module, tên file mã nguồn.
- Comment trong code. Không viết comment tiếng Việt.
- Docstring.
- Chuỗi log, thông báo lỗi, tên khóa trong dict và JSON.
- Nhãn trục, tiêu đề, legend của biểu đồ (xem mục 3).

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

Ngoại lệ duy nhất: tên file markdown, tên thư mục báo cáo, và nội dung file dữ liệu gốc tải từ Kaggle giữ nguyên như hiện có.

## 2. Văn phong tiếng Việt

- Văn phong học thuật, tự nhiên, viết như giáo trình hoặc bài giảng.
- Không dùng em dash. Thay bằng dấu phẩy, dấu hai chấm, dấu ngoặc đơn, hoặc tách thành câu riêng.
- Không dùng từ cường điệu: "cực kỳ", "siêu", "tuyệt vời", "bí quyết", "chìa khóa", "đột phá", "thần thánh", "must-know", "game changer".
- Không viết theo lối bán hàng hay thuyết phục. Không dùng các câu kiểu "nắm được điều này là nắm được nửa môn học", "chỉ cần 5 phút là hiểu", "đây là phần quan trọng nhất bạn không được bỏ qua". Trình bày nội dung, để người đọc tự đánh giá.
- Không dùng câu cảm thán và emoji trong phần nội dung. Emoji chỉ được dùng làm ký hiệu trong bảng trạng thái nếu thực sự cần.
- Diễn đạt dễ hiểu không có nghĩa là nói quá. Không hứa hẹn, không kết luận thiếu căn cứ.
- Mọi kết luận rút ra từ thí nghiệm phải kèm số liệu hoặc biểu đồ tương ứng. Nếu chưa có bằng chứng, viết rõ đó là dự đoán.

### 2.1 Viết thẳng vào nội dung, không dựng giàn giáo

Lỗi nặng nhất trong các bản nháp của thư mục này không phải lỗi từ vựng mà là lỗi cấu trúc: câu chỉ làm nhiệm vụ dẫn dắt, không mang thông tin, khiến người đọc phải đi qua hai ba câu mới gặp dữ kiện đầu tiên. Cấu trúc kiểu đó là dấu hiệu rõ nhất của văn bản do mô hình ngôn ngữ sinh ra. Các dạng cụ thể như sau.

**Câu bình luận về tầm quan trọng của chính nội dung sắp viết.** Không viết những câu chỉ thông báo rằng phần tiếp theo đáng chú ý.

> Sai: Backtracking cho phương pháp tăng tốc cần điều kiện chặt hơn. Đây là kết quả đáng chú ý nhất của mục này. Với $\alpha = 0{,}3$ và $t_0 = 1$, phương pháp không hội tụ, dừng ở sai số $5{,}5 \cdot 10^{-5}$.
>
> Đúng: Với $\alpha = 0{,}3$ và $t_0 = 1$, phương pháp tăng tốc không hội tụ mà dừng ở sai số $5{,}5 \cdot 10^{-5}$. Điều kiện Armijo thông thường không đủ cho phương pháp này.

Cùng loại và cùng phải bỏ: "Điểm cần làm rõ trước là", "Một tính chất cần nêu rõ ngay ở đây", "Con số đáng so sánh là", "Dưới đây là bốn phép thử quan trọng nhất", "Đây là lý do mục 8 có vai trò lớn hơn thông thường". Nếu một kết quả thực sự quan trọng thì số liệu của nó tự cho thấy điều đó, không cần người viết tuyên bố trước.

Quy tắc này không cấm câu chỉ đường. Một báo cáo dài vài chục trang cần câu nối giữa các mục, và câu chỉ đường mang thông tin thì hoàn toàn hợp lệ: "Mục 8 khai thác tính chất này", "Cách quy đổi hàm mục tiêu của thư viện nằm ở mục 7". Chúng cho biết nội dung gì nằm ở đâu, tức thêm một dữ kiện mà người đọc dùng được. Câu bị cấm là câu chỉ tuyên bố mức độ quan trọng mà không thêm dữ kiện nào, vì đánh giá tầm quan trọng thuộc về người đọc. Phép thử nhanh là xóa câu đó đi rồi xem người đọc có mất thông tin gì không.

**Tiêu đề `\paragraph` là nhãn, không phải câu khẳng định.** Đặt cả một mệnh đề có chủ ngữ vị ngữ làm tiêu đề là thói quen của bài viết kỹ thuật tiếng Anh trên blog, không phải của báo cáo học thuật tiếng Việt. Hệ quả thường thấy là câu đầu tiên của đoạn phải nhắc lại hoặc bình luận về tiêu đề, làm đoạn văn lửng lơ.

> Sai: `\paragraph{Newton vượt trội hoàn toàn trên bài toán này, và đó là điều hiển nhiên.}`
>
> Đúng: `\paragraph{Phương pháp Newton.}` rồi mở đầu đoạn bằng dữ kiện, chẳng hạn "Toàn bộ lời giải mất 0,076 giây, ít hơn 10 lần so với thời gian gradient descent cần để đạt $10^{-6}$."

**Không mở đầu câu bằng đại từ chỉ định trỏ về cả mệnh đề đứng trước.** Tiếng Anh cho phép "This is", "That means"; tiếng Việt thì "Đây là", "Điều này", "Điều đó" đứng đầu câu làm người đọc phải quay lại dò xem đại từ trỏ vào đâu. Hoặc gộp vào câu trước, hoặc nêu đích danh danh từ được trỏ.

> Sai: Lô kích thước 1 phân kỳ thành `NaN` ngay trong lượt duyệt đầu tiên. Đây là hệ quả trực tiếp của tỉ lệ 1159 lần nói trên.
>
> Đúng: Lô kích thước 1 phân kỳ thành `NaN` ngay trong lượt duyệt đầu tiên, đúng như tỉ lệ 1159 lần dự báo.

**Không lặp khung "Lý do là", "Nguyên nhân là".** Dùng liên từ nối thẳng vào mệnh đề chính bằng "vì", "do", "nên". Chỉ giữ khung này khi phần giải thích dài tới mức cần tách hẳn thành câu riêng.

**Chủ ngữ phải khớp với loại hành động.** Cài đặt thuật toán, chạy thí nghiệm, chọn tham số là việc của người làm, nên chủ ngữ phải là "chúng tôi", hoặc bỏ hẳn chủ ngữ. Viết "Chúng tôi cài đặt thêm ba phương pháp", hoặc "Ngoài bốn thuật toán bắt buộc, còn ba phương pháp nữa được cài đặt". Không viết "Báo cáo cài đặt thêm ba phương pháp", vì báo cáo không tự cài đặt được.

Khi câu nói về nội dung của chính văn bản thì lấy văn bản làm chủ ngữ lại đúng quy ước, cả trong tiếng Việt lẫn tiếng Anh. "Báo cáo gồm tám mục", "Mục 5 khảo sát tham số của từng thuật toán", "Bảng 3 cho thấy ba kết luận" đều dùng được. Ranh giới nằm ở chỗ văn bản trình bày và chứa đựng được, nhưng không cài đặt, không đo đạc và không quyết định được.

**Không kết đoạn bằng câu cách ngôn.** Câu kiểu "Tham số mặc định của thư viện không phải lúc nào cũng phù hợp với bài toán cụ thể" hay "Cận lý thuyết mô tả đúng phần đuôi nhưng nói rất ít về giai đoạn đầu" nghe như châm ngôn rút ra cho người đọc. Khi số liệu đã nêu ngay phía trên thì người đọc tự rút ra được. Nếu vẫn cần một câu tổng kết thì gắn nó với con số cụ thể, đừng phát biểu tổng quát.

**Không viết câu cụt thiếu vị ngữ.** "Ba kết luận.", "Hai cách xử lý, cả hai đều hội tụ.", "Ba nhận xét từ bảng 9." đều là cụm danh từ đứng một mình. Trong báo cáo học thuật tiếng Việt phải là câu đủ: "Từ bảng 3 có thể rút ra ba kết luận."

**Đa dạng nhịp câu.** Trong bản nháp gần nhất, 35 trên khoảng 242 câu kết thúc bằng dấu hai chấm, tức cứ bảy câu lại có một câu theo nhịp "khẳng định, hai chấm, giải thích". Lặp một khuôn cú pháp ở mật độ đó tạo cảm giác đều đều như máy. Dấu hai chấm chỉ nên dùng khi thật sự dẫn vào một liệt kê hoặc một định nghĩa.

**Không lặp công thức tương phản.** "Không phải X mà là Y", "không chỉ X mà còn Y", "X chứ không phải Y" đều là những khuôn hợp lệ, nhưng dùng dày sẽ lộ. Mỗi mục nên dùng nhiều nhất một lần.

### 2.2 Cấu trúc câu

- Câu dài quá 40 từ hoặc quá ba mệnh đề thì tách. Nối chuỗi mệnh đề bằng dấu phẩy là cách hành văn tiếng Anh, tiếng Việt đọc theo mạch ngắn hơn.
- Hạn chế bị động "được" khi đã có chủ thể rõ ràng. "Toàn bộ thí nghiệm được thực hiện trên một mẫu ngẫu nhiên" viết thành "Toàn bộ thí nghiệm chạy trên một mẫu ngẫu nhiên". Giữ "được" khi nó mang nghĩa khả năng, chẳng hạn "tính được", "đo được", "không phân biệt được".
- Không dùng bị động có tác nhân "bị chi phối bởi X". Đảo thành chủ động: "X chi phối".
- Không để giới từ treo cuối câu. "Trường hợp xấu nhất mà cận được thiết kế cho" phải viết là "trường hợp xấu nhất mà cận nhắm tới".
- Tiêu đề mục và tiểu mục dùng lối danh hóa hoặc động từ chủ động, không dùng bị động. Viết "Tạo biến tương tác trên cột đã chuẩn hóa", không viết "Biến tương tác phải được tạo trên cột đã chuẩn hóa".
- Kiểm tra các cụm dễ đọc nhầm do hai từ chức năng đứng cạnh nhau, chẳng hạn "hệ số quan sát được ước lượng bằng cách" hay "ba lịch trình giảm dần đều đưa sai số".
- Không chen từ tiếng Anh vào giữa câu tiếng Việt khi đã có từ tương đương: viết "trùng nhau từng bit", không viết "trùng nhau bit-by-bit". Thuật ngữ giữ nguyên tiếng Anh phải là thuật ngữ dùng nhất quán cả bài, chẳng hạn `line search`, `backtracking`, `epoch`.
- Không trộn văn nói vào báo cáo. Tránh "cho có", "thắng rõ", "thắng tuyệt đối", "đụng tới", "khó bị đánh bại", "đổi chác", "hội tụ thật", "chỉ đơn giản là".
- Không dùng hệ từ theo lối tiếng Anh: "cái giá phải trả là nhỏ" viết thành "cái giá phải trả không đáng kể"; "chi phí của bốn lịch trình là như nhau" viết thành "chi phí của bốn lịch trình như nhau".

### 2.3 Thuật ngữ và số

- Mỗi khái niệm dùng một tên duy nhất trong toàn bộ báo cáo. Nếu chọn "lượt duyệt dữ liệu" thì chú thích "(epoch)" ở lần xuất hiện đầu tiên rồi dùng thống nhất, không xen kẽ hai tên.
- Dấu thập phân trong văn xuôi tiếng Việt là dấu phẩy. Để `siunitx` in ra cùng dạng, `preamble.tex` khai báo `\sisetup{output-decimal-marker={,}}`. Số viết trực tiếp trong công thức toán giữ dấu chấm theo quy ước quốc tế.

## 3. Biểu đồ

- Vẽ bằng `matplotlib`. Được phép dùng `seaborn` cho phần tạo kiểu, hoặc thư viện Python khác nếu có lý do rõ ràng, nhưng mặc định là `matplotlib`.
- Toàn bộ chữ trên biểu đồ viết bằng tiếng Anh: nhãn trục, tiêu đề, legend, chú thích. Phần giải thích và kết luận về biểu đồ thì viết tiếng Việt trong văn bản kèm theo.
- Trục tung của biểu đồ hội tụ luôn là độ lớn hàm mục tiêu, dùng `semilogy` với đại lượng $f(w_k) - f^*$.
- Mỗi so sánh cần hai hình riêng: một theo số vòng lặp, một theo thời gian chạy tính bằng giây.
- Legend phải ghi rõ tham số, ví dụ `GD (t = 1/L)`, không ghi chung chung là `GD`.
- Mỗi thuật toán dùng một màu cố định xuyên suốt toàn bộ báo cáo. Định nghĩa bảng màu một lần trong `src/plotting.py` và gọi lại từ đó.
- Lưu mỗi hình ra `results/figures/` ở hai định dạng: PDF (dạng vector, để nhúng vào LaTeX) và PNG với `dpi=150` trở lên (để xem nhanh). Luôn dùng `bbox_inches='tight'`.
- Bản riêng cho slide, nếu cần bớt đường hoặc phóng to chữ so với bản trong báo cáo, lưu vào `results/figures/slides/` và giữ nguyên tên file. Lệnh `\resultgraphic` trong `preamble.tex` tìm ở thư mục này trước, không thấy mới lấy bản của báo cáo.
- Không dùng biểu đồ tương tác cho phần nộp bài, vì slide và báo cáo cần ảnh tĩnh.
- Kích thước hình đặt sao cho chữ trên hình đọc được khi chèn vào slide Beamer, thường là `figsize=(6, 4)` với cỡ chữ từ 10 trở lên. Không thu nhỏ hình trong LaTeX quá nhiều để bù cho hình vẽ quá to.

## 4. Công thức toán học

- Trong file markdown, viết công thức bằng LaTeX: `$...$` cho công thức trong dòng, `$$...$$` cho công thức tách khối.
- Không viết công thức toán trong khối code. Khối code chỉ dành cho mã nguồn thật sự chạy được, lệnh shell, hoặc cây thư mục.
- Giữ ký hiệu thống nhất với `KE_HOACH_TRIEN_KHAI.md`: $n$ số điểm dữ liệu, $d$ số thuộc tính, $X$ ma trận thiết kế, $y$ vector mục tiêu, $w$ tham số, $\lambda$ hệ số hiệu chỉnh, $t$ và $\eta$ độ dài bước, $L$ hằng số Lipschitz, $\mu$ hệ số lồi mạnh, $\kappa$ số điều kiện, $f^*$ giá trị tối ưu.

## 5. Báo cáo và slide

Báo cáo viết bằng LaTeX, slide làm bằng LaTeX Beamer. Không dùng Word, Google Docs, PowerPoint hay markdown cho hai sản phẩm này.

**Cấu trúc file.** Toàn bộ nằm trong `report/`:

- `report.tex`: báo cáo chính.
- `slides.tex`: slide Beamer.
- `refs.bib`: tài liệu tham khảo. Dùng `biblatex` với backend `biber`, không dùng `natbib`.
- `figures/`: chỉ chứa hình không sinh ra từ notebook, chẳng hạn sơ đồ vẽ tay. Hình kết quả thí nghiệm nằm nguyên ở `results/figures/` và được `\graphicspath` trong `preamble.tex` trỏ tới, không sao chép sang đây.
- `preamble.tex`: phần khai báo dùng chung cho cả hai file, tránh lặp.

**Xử lý tiếng Việt.** Biên dịch bằng XeLaTeX hoặc LuaLaTeX, không dùng pdfLaTeX, vì cần font Unicode đầy đủ dấu tiếng Việt. Khai báo đã kiểm chứng chạy được trên máy này (TeX Live 2025, macOS):

```latex
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{vietnamese}
\setmainfont{texgyretermes}[
  Extension    = .otf,
  UprightFont  = *-regular,
  BoldFont     = *-bold,
  ItalicFont   = *-italic,
  BoldItalicFont = *-bolditalic,
]
```

Lưu ý: gọi font theo tên hiển thị (`\setmainfont{TeX Gyre Termes}`) sẽ báo lỗi không tìm thấy, phải gọi theo tên file như trên. Hai phương án thay thế đã kiểm chứng: `\setmainfont{Times New Roman}` (font hệ thống macOS), hoặc bỏ hẳn dòng `\setmainfont` để dùng Latin Modern mặc định của `fontspec`.

Biên dịch bằng `latexmk -xelatex report.tex` để tự xử lý số lần chạy và phần tài liệu tham khảo.

**Quy ước trong báo cáo.**

- Ký hiệu toán học thống nhất với mục 4 và với `KE_HOACH_TRIEN_KHAI.md`.
- Công thức cần được đánh số nếu có tham chiếu tới, dùng `\eqref` chứ không viết "công thức ở trên".
- Mọi hình và bảng phải có `\caption`, có `\label`, và được nhắc tới ít nhất một lần trong phần thân bằng `\ref`. Không chèn cùng một file hình hai lần dưới hai nhãn khác nhau.
- Quy tắc trên từng bị vi phạm ở 6 trên 10 bảng và ở 18 trên 22 hình khi chỉ kiểm tra bằng mắt, nên nó được kiểm tra tự động trong `tests/test_report.py`. Chạy `pytest tests/test_report.py` trước mỗi lần nộp. Bài kiểm tra đối chiếu `\label` với `\ref`, bắt nhãn trùng, bắt hình bị chèn hai lần, kiểm tra file hình có thật trong `results/figures/`, và kiểm tra mọi mục trong `refs.bib` đều được trích dẫn.
- Mọi phát biểu lấy từ tài liệu, gồm định lý, điều kiện hội tụ, tốc độ hội tụ và công thức tham số của một thuật toán, phải trích dẫn nguồn bằng `\cite`. Số liệu tự đo thì không trích dẫn. Mục nào trong `refs.bib` không được trích dẫn thì xóa đi, vì `biblatex` kiểu `numeric` bỏ qua mục đó và in ra danh mục rỗng mà không báo lỗi.
- Nhúng hình bằng `\includegraphics` với file PDF, đặt độ rộng theo `\linewidth` chứ không đặt kích thước tuyệt đối.
- Thuật toán trình bày bằng `algorithm` kết hợp `algpseudocode`, viết bằng tiếng Anh theo quy tắc mục 1. Không dán mã Python nguyên khối vào báo cáo, chỉ trích đoạn ngắn bằng `listings` hoặc `minted` khi thật sự cần.
- Bảng dùng `booktabs`. Không dùng đường kẻ dọc.

**Quy ước trong slide Beamer.**

- Chọn theme đơn giản. Slide hiện dùng theme `default` với thanh tiêu đề màu `darkblue`, theo mẫu chung với báo cáo môn Toán rời rạc, nên không cần cài thêm gói nào. Nếu đổi sang theme cần cài thêm, phải ghi rõ trong `report/README.md` cách cài, để mọi thành viên biên dịch được.
- Mỗi frame một ý. Slide biểu đồ chỉ chứa hình và tối đa hai dòng kết luận, phần diễn giải dài để người trình bày nói.
- Dùng `\note{}` cho ghi chú người trình bày, không nhét vào phần hiển thị.
- Cấu trúc frame bám theo outline ở mục 10 của `KE_HOACH_TRIEN_KHAI.md`.
- Không dùng hiệu ứng chuyển slide. `\pause` và `\onslide` chỉ dùng khi thật sự cần bộc lộ nội dung theo trình tự.

**Sản phẩm biên dịch.** Không đưa file do trình biên dịch sinh ra vào git. Danh sách đầy đủ nằm trong `.gitignore`, gồm `.pdf` và các file trung gian `.aux`, `.log`, `.out`, `.nav`, `.snm`, `.toc`, `.bbl`, `.bcf`, `.blg`, `.fls`, `.fdb_latexmk`, `.run.xml`, `.synctex.gz`, `.vrb`, `.xdv`. Khi thêm gói LaTeX sinh ra đuôi file khác, cập nhật `.gitignore` ngay trong lần commit đó.

## 6. Tổ chức mã nguồn

- Logic thuật toán đặt trong `src/`. Notebook chỉ gọi hàm, chạy thí nghiệm và vẽ hình, không định nghĩa lại thuật toán.
- Mọi thuật toán tối ưu hóa dùng chung một chữ ký hàm và trả về cùng một cấu trúc lịch sử, theo mô tả ở mục 4.1 của kế hoạch.
- Khi đo thời gian chạy, dừng đồng hồ trước khi tính và ghi log, chạy lại sau đó. Thời gian ghi log không được tính vào thời gian thuật toán.
- Cố định seed cho mọi thành phần ngẫu nhiên và ghi seed vào kết quả.
- Kết quả chạy lưu ra `results/raw/` dạng JSON để vẽ lại được mà không cần chạy lại thí nghiệm.

## 7. Phạm vi

Trọng tâm của bài tập là tối ưu hóa. Không mở rộng sang feature engineering phức tạp, thử nhiều họ mô hình, hay tinh chỉnh siêu tham số ngoài phạm vi đã nêu trong kế hoạch, trừ khi được yêu cầu rõ ràng.
