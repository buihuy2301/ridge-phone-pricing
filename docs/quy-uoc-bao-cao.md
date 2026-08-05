# Quy ước cho báo cáo và slide

File này là bản đầy đủ của mục 5 trong `CLAUDE.md`. Đọc trước khi sửa
`report/report.tex`, `report/slides.tex` hoặc `report/preamble.tex`. Phần biên
dịch, danh sách gói và mẫu trình bày nằm ở `report/README.md`, không lặp lại ở
đây.

Báo cáo viết bằng LaTeX, slide làm bằng LaTeX Beamer. Không dùng Word, Google
Docs, PowerPoint hay markdown cho hai sản phẩm này.

## Cấu trúc file

Toàn bộ nằm trong `report/`:

- `report.tex`: báo cáo chính.
- `slides.tex`: slide Beamer.
- `refs.bib`: tài liệu tham khảo. Dùng `biblatex` với backend `biber`, không dùng
  `natbib`.
- `figures/`: chỉ chứa hình không sinh ra từ notebook, chẳng hạn sơ đồ vẽ tay.
  Hình kết quả thí nghiệm nằm nguyên ở `results/figures/` và được `\graphicspath`
  trong `preamble.tex` trỏ tới, không sao chép sang đây.
- `preamble.tex`: phần khai báo dùng chung cho cả hai file, tránh lặp.

## Xử lý tiếng Việt

Biên dịch bằng XeLaTeX hoặc LuaLaTeX, không dùng pdfLaTeX, vì cần font Unicode
đầy đủ dấu tiếng Việt. Khai báo đã kiểm chứng chạy được trên máy này (TeX Live
2025, macOS):

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

Lưu ý: gọi font theo tên hiển thị (`\setmainfont{TeX Gyre Termes}`) sẽ báo lỗi
không tìm thấy, phải gọi theo tên file như trên. Hai phương án thay thế đã kiểm
chứng: `\setmainfont{Times New Roman}` (font hệ thống macOS), hoặc bỏ hẳn dòng
`\setmainfont` để dùng Latin Modern mặc định của `fontspec`.

Biên dịch bằng `latexmk -xelatex report.tex` để tự xử lý số lần chạy và phần tài
liệu tham khảo.

## Quy ước trong báo cáo

- Ký hiệu toán học thống nhất với mục 4 của `CLAUDE.md` và với
  `KE_HOACH_TRIEN_KHAI.md`.
- Công thức cần được đánh số nếu có tham chiếu tới, dùng `\eqref` chứ không viết
  "công thức ở trên".
- Mọi hình và bảng phải có `\caption`, có `\label`, và được nhắc tới ít nhất một
  lần trong phần thân bằng `\ref`. Không chèn cùng một file hình hai lần dưới hai
  nhãn khác nhau.
- Quy tắc trên từng bị vi phạm ở 6 trên 10 bảng và ở 18 trên 22 hình khi chỉ kiểm
  tra bằng mắt, nên nó được kiểm tra tự động trong `tests/test_report.py`. Chạy
  `pytest tests/test_report.py` trước mỗi lần nộp. Bài kiểm tra đối chiếu `\label`
  với `\ref`, bắt nhãn trùng, bắt hình bị chèn hai lần, kiểm tra file hình có thật
  trong `results/figures/`, và kiểm tra mọi mục trong `refs.bib` đều được trích
  dẫn.
- Mọi phát biểu lấy từ tài liệu, gồm định lý, điều kiện hội tụ, tốc độ hội tụ và
  công thức tham số của một thuật toán, phải trích dẫn nguồn bằng `\cite`. Số liệu
  tự đo thì không trích dẫn. Mục nào trong `refs.bib` không được trích dẫn thì xóa
  đi, vì `biblatex` kiểu `numeric` bỏ qua mục đó và in ra danh mục rỗng mà không
  báo lỗi.
- Nhúng hình bằng `\includegraphics` với file PDF, đặt độ rộng theo `\linewidth`
  chứ không đặt kích thước tuyệt đối.
- Thuật toán trình bày bằng `algorithm` kết hợp `algpseudocode`, viết bằng tiếng
  Anh theo quy tắc mục 1 của `CLAUDE.md`. Không dán mã Python nguyên khối vào báo
  cáo, chỉ trích đoạn ngắn bằng `listings` hoặc `minted` khi thật sự cần.
- Bảng dùng `booktabs`. Không dùng đường kẻ dọc.
- Tiêu đề chương in một dòng theo dạng `VII. So sánh với thư viện`, cỡ `\Large`,
  số chương bằng chữ số La Mã qua `titlesec`. Mục, hình, bảng, công thức và
  thuật toán vẫn lấy tiền tố chương bằng chữ số Ả Rập, tức `mục 7.1` và
  `hình 7.2`. Ô số chương trong mục lục nới lên `2.8em` bằng `tocloft`, vì
  `VIII` không vừa bề rộng mặc định.
- Hình và bảng đặt `[tbp]`, không dùng `h` hay `H`. Với `h`, một hình khai báo
  giữa hai đoạn của cùng một mạch lập luận sẽ chèn ngay vào đó và cắt đôi lập
  luận; bỏ `h` thì hình lên đầu hoặc xuống cuối trang và phần văn xuôi liền
  mạch.

## Quy ước trong slide Beamer

- Chọn theme đơn giản. Slide hiện dùng theme `default` với thanh tiêu đề màu
  `darkblue`, theo mẫu chung với báo cáo môn Toán rời rạc, nên không cần cài thêm
  gói nào. Nếu đổi sang theme cần cài thêm, phải ghi rõ trong `report/README.md`
  cách cài, để mọi thành viên biên dịch được.
- Mỗi frame một ý. Slide biểu đồ chỉ chứa hình và tối đa hai dòng kết luận, phần
  diễn giải dài để người trình bày nói.
- Dùng `\note{}` cho ghi chú người trình bày, không nhét vào phần hiển thị.
- Cấu trúc frame bám theo outline ở mục 10 của `KE_HOACH_TRIEN_KHAI.md`.
- Không dùng hiệu ứng chuyển slide. `\pause` và `\onslide` chỉ dùng khi thật sự
  cần bộc lộ nội dung theo trình tự.

## Sản phẩm biên dịch

Không đưa file do trình biên dịch sinh ra vào git. Danh sách đầy đủ nằm trong
`.gitignore`, gồm `.pdf` và các file trung gian `.aux`, `.log`, `.out`, `.nav`,
`.snm`, `.toc`, `.bbl`, `.bcf`, `.blg`, `.fls`, `.fdb_latexmk`, `.run.xml`,
`.synctex.gz`, `.vrb`, `.xdv`. Khi thêm gói LaTeX sinh ra đuôi file khác, cập nhật
`.gitignore` ngay trong lần commit đó.
