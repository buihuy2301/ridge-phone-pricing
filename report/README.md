# Biên dịch báo cáo và slide

Báo cáo và slide đều viết bằng LaTeX, biên dịch bằng XeLaTeX vì cần font Unicode
đầy đủ dấu tiếng Việt.

## Yêu cầu

TeX Live hoặc MacTeX bản đầy đủ. Các gói được dùng: `fontspec`, `polyglossia`,
`amsmath`, `mathtools`, `graphicx`, `booktabs`, `siunitx`, `array`, `float`,
`fancyhdr`, `setspace`, `tikz`, `colortbl`, `algorithm`, `algpseudocode`,
`listings`, `url`, `hyperref`, `biblatex` kèm `biber`. Slide dùng theme Beamer
`default`, không cần cài thêm gì. Tất cả đều có sẵn trong bản TeX Live đầy đủ.

Kiểm tra nhanh một gói:

```bash
kpsewhich fancyhdr.sty
```

## Trình bày

Báo cáo và slide dùng chung mẫu trình bày với báo cáo môn Toán rời rạc:

- Báo cáo dùng lớp `report`, mỗi mục lớn là một chương bắt đầu ở trang mới. Trang
  bìa lấy thông tin từ các macro `\coursename`, `\reportname`, `\reporttitle`,
  `\studentname`, `\teachername` khai báo ở đầu `report.tex`; đổi tên đề tài hay
  danh sách thành viên thì sửa ở đó, không sửa rải rác trong phần thân.
- Đầu trang ghi tên báo cáo bên trái, tên trường và tên môn bên phải; chân trang
  ghi số trang ở góc phải. Đoạn văn không thụt đầu dòng, giãn dòng 1,5.
- Slide dùng theme `default` với thanh tiêu đề màu `darkblue`, tỉ lệ 16:9, logo
  trường ở góc dưới bên phải và số trang ở chân slide.
- Logo trường nằm ở `report/figures/logo-hus.jpg`.
- `slides-notes.tex` là bản slide kèm kịch bản nói. Nó không chứa nội dung riêng:
  chỉ đặt cờ `\shownotes` rồi `\input{slides.tex}`, và cờ đó bật khối
  `\setbeameroption{show notes}` ở đầu `slides.tex`. Sản phẩm là
  `slides-notes.pdf`, xen kẽ mỗi slide với một trang mang ảnh thu nhỏ của slide
  đó cùng phần `\note{}` viết cho nó. Vì hai file dùng chung một nguồn, sửa slide
  mà quên sửa kịch bản thì thấy ngay ở lần biên dịch sau. Bản chiếu
  `slides.pdf` không đổi khi không có cờ.

## Biên dịch

Chạy từ trong thư mục `report/`:

```bash
latexmk -xelatex report.tex
latexmk -xelatex slides.tex
latexmk -xelatex slides-notes.tex
```

Dọn file trung gian:

```bash
latexmk -C
```

## Hình

Hai file `.tex` lấy hình từ `../results/figures/`, do notebook sinh ra dưới dạng
PDF. Hình nào chưa được sinh sẽ hiện thành một khung trống có ghi tên file, nên
tài liệu vẫn biên dịch được khi thí nghiệm chưa chạy xong. Sau khi chạy notebook,
biên dịch lại là hình tự xuất hiện.

Nếu cần một hình riêng cho báo cáo mà không sinh từ thí nghiệm, đặt vào
`report/figures/`.

## Font

`preamble.tex` khai báo font theo tên file (`texgyretermes`), không theo tên hiển
thị, vì gọi `\setmainfont{TeX Gyre Termes}` báo lỗi không tìm thấy trên máy đã
kiểm thử. Hai phương án thay thế đã kiểm chứng: `\setmainfont{Times New Roman}`,
hoặc bỏ hẳn dòng `\setmainfont` để dùng Latin Modern mặc định của `fontspec`.
