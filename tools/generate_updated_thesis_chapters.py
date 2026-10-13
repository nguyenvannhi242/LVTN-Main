from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_updates"


BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK_BLUE = RGBColor(0x1F, 0x4D, 0x78)
GRAY = RGBColor(0x66, 0x66, 0x66)
BLACK = RGBColor(0x00, 0x00, 0x00)


@dataclass
class TableSpec:
    insert_after: str
    title: str
    headers: list[str]
    rows: list[list[str]]
    widths: list[float]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_page_layout(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)


def ensure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)

    p = normal.paragraph_format
    p.space_before = Pt(0)
    p.space_after = Pt(8)
    p.line_spacing = 1.333
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        fmt = style.paragraph_format
        fmt.space_before = Pt(before)
        fmt.space_after = Pt(after)
        fmt.line_spacing = 1.15

    if "Body Tight" not in doc.styles:
        style = doc.styles.add_style("Body Tight", WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = doc.styles["Normal"]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(11)
        fmt = style.paragraph_format
        fmt.space_after = Pt(4)
        fmt.line_spacing = 1.208
        fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_title_page(doc: Document, chapter_title: str, subtitle: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("BÁO CÁO LUẬN VĂN TỐT NGHIỆP")
    r.bold = True
    r.font.name = "Calibri"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    r.font.size = Pt(14)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(chapter_title)
    r.bold = True
    r.font.name = "Calibri"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    r.font.size = Pt(20)
    r.font.color.rgb = BLACK

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(subtitle)
    r.italic = True
    r.font.name = "Calibri"
    r._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    r._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    r.font.size = Pt(11)
    r.font.color.rgb = GRAY

    meta_rows = [
        ("Đề tài", "Xây dựng hệ thống phát hiện mã nguồn C++ do AI sinh ra"),
        ("Mục đích", "Phiên bản chương đã rà soát và cập nhật theo hiện trạng repo"),
        ("Nguồn đối chiếu", "README.md, ARCHITECTURE_MAP.md, mã nguồn trong thư mục src/"),
        ("Ngày tạo", "20/05/2026"),
    ]
    for label, value in meta_rows:
        p = doc.add_paragraph(style="Body Tight")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        label_run = p.add_run(f"{label}: ")
        label_run.bold = True
        value_run = p.add_run(value)
        for run in (label_run, value_run):
            run.font.name = "Calibri"
            run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
            run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
            run.font.size = Pt(11)

    doc.add_section(WD_SECTION_START.NEW_PAGE)


def add_heading(doc: Document, text: str, level: int) -> None:
    doc.add_paragraph(text, style=f"Heading {level}")


def add_paragraph(doc: Document, text: str, style: str = "Normal") -> None:
    p = doc.add_paragraph(style=style)
    p.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="Body Tight")
        p.paragraph_format.left_indent = Inches(0.375)
        p.paragraph_format.first_line_indent = Inches(-0.194)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.208
        p.add_run(f"• {item}")


def add_table(doc: Document, spec: TableSpec) -> None:
    title = doc.add_paragraph(style="Body Tight")
    title.paragraph_format.space_before = Pt(4)
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run(spec.title)
    run.bold = True

    table = doc.add_table(rows=1, cols=len(spec.headers))
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    table.style = "Table Grid"

    for idx, header in enumerate(spec.headers):
        cell = table.rows[0].cells[idx]
        cell.width = Inches(spec.widths[idx])
        cell.text = header
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, "F2F4F7")
        set_cell_margins(cell)
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in para.runs:
            run.bold = True
            run.font.name = "Calibri"
            run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
            run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
            run.font.size = Pt(10.5)

    for row_data in spec.rows:
        row = table.add_row()
        for idx, value in enumerate(row_data):
            cell = row.cells[idx]
            cell.width = Inches(spec.widths[idx])
            cell.text = value
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.JUSTIFY
            for run in para.runs:
                run.font.name = "Calibri"
                run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
                run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
                run.font.size = Pt(10.5)

    doc.add_paragraph("", style="Body Tight")


def chapter_2_content() -> tuple[str, str, list[tuple[str, int]], list[TableSpec]]:
    parts = [
        ("CHƯƠNG 2: NỀN TẢNG LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN", 1),
        (
            "Chương này trình bày cơ sở học thuật cho bài toán phát hiện mã nguồn C++ do AI sinh ra, đồng thời hiệu chỉnh lại góc nhìn lý thuyết để phù hợp với hướng triển khai hiện tại của đề tài. Trọng tâm không chỉ nằm ở mô hình phân loại mã nguồn, mà còn bao gồm khả năng giải thích, suy luận nhiều tín hiệu và khả năng triển khai hệ thống ở mức ứng dụng.",
            0,
        ),
        ("2.1 Cơ sở lý thuyết", 2),
        ("2.1.1 Bài toán phát hiện mã nguồn do AI tạo ra", 3),
        (
            "Bài toán AI code detection được phát biểu như một bài toán phân loại nhị phân: với một đoạn mã C++ đầu vào, hệ thống cần xác định mã đó gần với phong cách do con người viết hay được sinh bởi mô hình AI. Trong bối cảnh học thuật, bài toán này gắn chặt với yêu cầu bảo đảm tính liêm chính học tập, vì mã do AI sinh ra thường không bị phát hiện bằng các công cụ đạo văn truyền thống vốn dựa vào so khớp chuỗi hoặc cây cú pháp.",
            0,
        ),
        (
            "Khó khăn cốt lõi của bài toán nằm ở chỗ mã AI không phải bản sao trực tiếp của một nguồn có sẵn, mà thường là một phiên bản mới, hợp lệ về cú pháp, sạch về định dạng và nhất quán về phong cách. Vì vậy, hệ thống phát hiện phải dựa trên các tín hiệu phong cách, cấu trúc, xác suất ngôn ngữ và dấu vết hình thái học của mã nguồn.",
            0,
        ),
        ("2.1.2 Mô hình Transformer và biểu diễn mã nguồn", 3),
        (
            "Transformer là kiến trúc nền tảng của hầu hết các mô hình ngôn ngữ hiện đại nhờ cơ chế self-attention, cho phép mô hình học phụ thuộc dài hạn trong chuỗi đầu vào. Khi áp dụng vào mã nguồn, Transformer không chỉ nắm bắt được quan hệ tuần tự giữa token mà còn có thể học các mẫu định danh, cú pháp và cấu trúc lặp lại vốn phản ánh thói quen lập trình.",
            0,
        ),
        (
            "Trong họ mô hình dùng cho mã nguồn, BERT và RoBERTa cung cấp nền tảng encoder mạnh cho các tác vụ phân loại. CodeBERT mở rộng hướng này cho dữ liệu song ngữ mã nguồn và ngôn ngữ tự nhiên, còn GraphCodeBERT tiến thêm một bước bằng cách đưa thông tin data flow vào quá trình tiền huấn luyện. Điều này đặc biệt phù hợp với mã C++, nơi quan hệ giữa biến, hàm và lớp có vai trò lớn trong việc thể hiện phong cách hiện thực lời giải.",
            0,
        ),
        (
            "Đối với đề tài, GraphCodeBERT có ý nghĩa theo hai lớp. Thứ nhất, đây là mô hình có năng lực biểu diễn tốt cho mã nguồn, thích hợp để xây dựng bộ phân loại học sâu. Thứ hai, việc mô hình chú ý tới cấu trúc luồng dữ liệu giúp giảm phụ thuộc vào tín hiệu bề mặt thuần túy, nhờ đó phù hợp hơn với mục tiêu phân biệt mã do AI sinh ra và mã do con người viết trong các bài toán C++ có cấu trúc không đồng nhất.",
            0,
        ),
        ("2.1.3 Giải thích mô hình trong phát hiện mã AI", 3),
        (
            "Một bộ phân loại có độ chính xác cao nhưng không giải thích được vẫn khó được chấp nhận trong môi trường học thuật. Giảng viên và người đánh giá không chỉ cần nhãn kết luận, mà còn cần biết vì sao hệ thống nghiêng về phía AI hay con người. Vì vậy, Explainable AI là thành phần học thuật và thực tiễn không thể tách rời trong đề tài.",
            0,
        ),
        (
            "Về nguyên lý, có thể chia phương pháp giải thích thành hai nhóm: nhóm nội tại, trong đó cấu trúc mô hình đã có khả năng diễn giải, và nhóm hậu nghiệm, trong đó một kỹ thuật riêng được áp dụng lên mô hình đã huấn luyện. Với đề tài này, hai kiểu giải thích có giá trị nhất là giải thích theo token và giải thích theo đặc trưng. Giải thích theo token cho biết đoạn mã nào kéo quyết định của mô hình học sâu; giải thích theo đặc trưng cho biết những dấu hiệu phong cách nào đang chi phối kết quả phân loại.",
            0,
        ),
        (
            "Trên phương diện thực thi, Layer Integrated Gradients phù hợp để xây dựng bản đồ nhiệt ở cấp token, trong khi SHAP phù hợp với mô hình đặc trưng như LightGBM để diễn giải đóng góp của từng chỉ số hình thái. Cách tiếp cận kết hợp này giúp mở rộng lời giải từ một bộ phân loại học sâu thuần túy thành một hệ thống phân tích có thể kiểm chứng.",
            0,
        ),
        ("2.1.4 Perplexity, stylometry và suy luận đa tín hiệu", 3),
        (
            "Perplexity phản ánh mức độ dễ dự đoán của chuỗi mã nguồn đối với một mô hình ngôn ngữ sinh. Giả thuyết thường gặp là mã do AI sinh ra có xu hướng đều hơn, sạch hơn và ít bất ngờ hơn, nên mức perplexity có thể thấp hơn so với mã do con người viết. Tuy nhiên, đây không phải tín hiệu đủ mạnh để dùng độc lập, vì nhiều trường hợp mã của sinh viên cũng ngắn, chuẩn hóa và dễ dự đoán.",
            0,
        ),
        (
            "Stylometry nguồn mã, tức phân tích phong cách lập trình thông qua các đặc trưng như độ dài dòng, mật độ comment, quy ước đặt tên biến, độ sâu lồng nhau hay entropy khoảng trắng, lại cho thấy một góc nhìn khác. Các đặc trưng này vốn được sử dụng nhiều trong bài toán nhận diện tác giả mã nguồn, và có thể tái sử dụng như nền tảng cho việc phân biệt phong cách AI với phong cách con người.",
            0,
        ),
        (
            "Do mỗi tín hiệu đều có giới hạn, hướng phù hợp về mặt lý thuyết là suy luận đa tín hiệu. Một hệ thống đáng tin cậy nên biết kết hợp đầu ra phân loại, giải thích token, đặc trưng stylometry, perplexity và luật ra quyết định để giảm phụ thuộc vào một chỉ báo đơn lẻ.",
            0,
        ),
        ("2.1.5 Cơ sở lý thuyết cho hệ thống agent và kiến trúc triển khai", 3),
        (
            "Ngoài phần mô hình, đề tài còn có bản chất của một hệ thống phần mềm AI nhiều thành phần. Khi đó, lý thuyết liên quan không chỉ là học sâu mà còn là orchestration, phân tầng dịch vụ và quản trị dữ liệu. Ý tưởng agent ở đây không nên hiểu như một tác nhân tự trị tổng quát, mà là một workflow có trạng thái, trong đó các nút xử lý được tách vai trò rõ ràng: định tuyến, phân tích, đánh giá và sinh giải thích.",
            0,
        ),
        (
            "Song song với đó, kiến trúc dữ liệu nhiều tầng như Bronze, Silver, Gold có vai trò quan trọng trong việc tách dữ liệu thô, dữ liệu đặc trưng và kết quả suy luận. Đây là cơ sở lý thuyết cho việc mở rộng đề tài từ một notebook nghiên cứu sang một hệ thống có khả năng lưu vết, giám sát và tái huấn luyện.",
            0,
        ),
        ("2.2 Các công trình nghiên cứu liên quan", 2),
        ("2.2.1 Nghiên cứu về phát hiện mã do AI sinh ra", 3),
        (
            "Các công trình gần đây về AI-generated code detection chủ yếu tập trung vào hai hướng. Hướng thứ nhất dùng embedding học sâu hoặc biểu diễn trích xuất từ mô hình tiền huấn luyện để huấn luyện bộ phân loại nhị phân. Hướng thứ hai dùng đặc trưng hình thái hoặc xác suất để phát hiện sự khác biệt trong phong cách mã nguồn. Phần lớn các nghiên cứu này chứng minh bài toán có thể giải được, nhưng thường chỉ khảo sát trên một số ngôn ngữ phổ biến như Python hoặc Java, hoặc chỉ gói trong môi trường thực nghiệm hẹp.",
            0,
        ),
        ("2.2.2 Nghiên cứu về nhận diện tác giả mã nguồn", 3),
        (
            "Nhận diện tác giả mã nguồn không trùng hoàn toàn với AI code detection, nhưng đây là hướng nền tảng về phương pháp luận. Nếu một hệ thống có thể nhận ra dấu vết phong cách cá nhân của từng lập trình viên, thì về nguyên tắc nó cũng có thể học để nhận ra dấu vết phong cách mang tính thống kê của mã do AI tạo ra. Do đó, các công trình về stylometry, character n-gram, TF-IDF, CNN, RNN và Transformer cho authorship attribution vẫn có giá trị tham chiếu mạnh cho đề tài.",
            0,
        ),
        ("2.2.3 Nghiên cứu về biểu diễn mã nguồn và benchmark", 3),
        (
            "Các họ mô hình như CodeBERT, GraphCodeBERT, các biến thể dựa trên AST và graph neural networks đã cho thấy biểu diễn mã nguồn không nên bị thu hẹp vào chuỗi token thuần túy. Dù vậy, các benchmark phổ biến như CodeXGLUE chủ yếu phục vụ các tác vụ hiểu mã tổng quát, chưa cung cấp tập chuẩn dành riêng cho C++ AI code detection. Khoảng thiếu này buộc đề tài phải tự xây dựng dữ liệu và đồng thời làm cho kết quả nghiên cứu có giá trị ứng dụng riêng.",
            0,
        ),
        ("2.2.4 Khoảng trống nghiên cứu", 3),
        (
            "Qua tổng hợp các hướng nghiên cứu liên quan, có thể rút ra bốn khoảng trống chính làm cơ sở cho đề tài. Thứ nhất, còn thiếu tập dữ liệu C++ chuyên biệt cho bài toán phát hiện mã do AI sinh ra, đặc biệt là dữ liệu có phân biệt phong cách Basic và OOP. Thứ hai, nhiều hệ thống chỉ dừng ở việc đưa ra nhãn dự đoán mà chưa giải thích rõ ràng ở cấp token và cấp đặc trưng. Thứ ba, chưa nhiều công trình tích hợp nhiều tín hiệu trong cùng một pipeline suy luận có kiểm tra chéo. Thứ tư, khoảng cách giữa prototype nghiên cứu và hệ thống triển khai thực tế vẫn còn lớn; các yếu tố như orchestration, lưu trữ dữ liệu, dashboard và background processing thường bị tách khỏi phần mô hình.",
            0,
        ),
        ("2.3 Tổng kết chương", 2),
        (
            "Từ cơ sở lý thuyết đã phân tích, có thể thấy lời giải phù hợp cho bài toán này không nên chỉ là một bộ phân loại đơn lẻ. Đề tài cần một hướng tiếp cận kết hợp biểu diễn mã nguồn bằng Transformer, giải thích bằng XAI, bổ trợ bằng stylometry và perplexity, đồng thời đặt toàn bộ quá trình vào một kiến trúc hệ thống có khả năng vận hành và mở rộng. Đây chính là cơ sở để xây dựng giải pháp đề xuất trong Chương 3.",
            0,
        ),
    ]

    tables = [
        TableSpec(
            insert_after="2.1.5 Cơ sở lý thuyết cho hệ thống agent và kiến trúc triển khai",
            title="Bảng 2.1. So sánh các nhóm tín hiệu dùng trong phát hiện mã nguồn do AI sinh ra",
            headers=["Nhóm tín hiệu", "Ý nghĩa", "Giới hạn khi dùng đơn lẻ"],
            widths=[1.6, 2.5, 2.4],
            rows=[
                ["Phân loại học sâu", "Học mẫu ngôn ngữ và cấu trúc mã nguồn từ dữ liệu lớn.", "Khó giải thích nếu chỉ trả về xác suất."],
                ["Stylometry / đặc trưng tĩnh", "Phản ánh thói quen đặt tên, định dạng và cấu trúc lệnh.", "Dễ bị nhiễu nếu người viết cố tình chuẩn hóa phong cách."],
                ["Perplexity", "Đo độ dễ dự đoán của chuỗi mã đối với mô hình sinh.", "Không đủ mạnh nếu tách rời ngữ cảnh và loại bài toán."],
                ["XAI", "Làm rõ token hoặc đặc trưng nào đang kéo kết quả.", "Phụ thuộc vào loại mô hình và chi phí tính toán."],
            ],
        ),
        TableSpec(
            insert_after="2.2.4 Khoảng trống nghiên cứu",
            title="Bảng 2.2. Các khoảng trống nghiên cứu được kế thừa trong đề tài",
            headers=["Khoảng trống", "Biểu hiện trong tài liệu liên quan", "Hướng giải quyết của đề tài"],
            widths=[1.5, 2.6, 2.4],
            rows=[
                ["Thiếu dữ liệu C++ chuyên biệt", "Ít benchmark trực tiếp cho C++ AI code detection.", "Tự xây dựng dữ liệu và tách Basic/OOP."],
                ["Thiếu khả năng giải thích", "Nhiều nghiên cứu chỉ đưa ra nhãn hoặc accuracy.", "Kết hợp giải thích cấp token và cấp đặc trưng."],
                ["Thiếu suy luận đa tín hiệu", "Một số hệ thống chỉ dùng một bộ phân loại duy nhất.", "Kết hợp DL, stylometry, perplexity và luật đánh giá."],
                ["Thiếu góc nhìn hệ thống", "Phần triển khai thường không đi cùng phần mô hình.", "Thiết kế kiến trúc nhiều dịch vụ và dữ liệu nhiều tầng."],
            ],
        ),
    ]
    return "Chương 2 cập nhật", "Nền tảng lý thuyết và công trình liên quan", parts, tables


def chapter_3_content() -> tuple[str, str, list[tuple[str, int]], list[TableSpec]]:
    parts = [
        ("CHƯƠNG 3: GIẢI PHÁP ĐỀ XUẤT", 1),
        (
            "Chương này mô tả lại giải pháp theo đúng trạng thái phát triển của đề tài sau nhiều vòng thay đổi. Thay vì xem toàn bộ hệ thống như một phiên bản duy nhất, chương được tổ chức theo hai lớp: lớp nghiên cứu mô hình và lớp triển khai hệ thống. Cách trình bày này giúp đồng bộ hơn với repo hiện tại và làm rõ đâu là phần cốt lõi cần bảo vệ trong luận văn.",
            0,
        ),
        ("3.1 Định hướng kiến trúc và phạm vi giải pháp", 2),
        ("3.1.1 Hai giai đoạn phát triển của đề tài", 3),
        (
            "Giai đoạn đầu của đề tài tập trung vào lời giải nghiên cứu: xây dựng dữ liệu, fine-tune GraphCodeBERT, xây một workflow tác nhân cho suy luận và triển khai thử nghiệm giao diện phân tích trực quan bằng Streamlit kết nối tới dịch vụ GPU chạy trên Google Colab. Đây là phiên bản phù hợp để chứng minh tính khả thi của ý tưởng.",
            0,
        ),
        (
            "Giai đoạn sau mở rộng đề tài thành một hệ thống có khả năng vận hành: giao diện người dùng chuyển sang Django, lớp AI được tách thành FastAPI service, dữ liệu được tổ chức theo Medallion architecture, tác vụ nền dùng Celery, còn phần suy luận nặng có thể offload sang local GPU server hoặc Colab runtime. Đây là hướng triển khai phản ánh rõ hơn yêu cầu của một hệ thống phân tích mã nguồn thực tế.",
            0,
        ),
        ("3.1.2 Phạm vi chính thức được sử dụng trong báo cáo cập nhật", 3),
        (
            "Trong bản cập nhật này, lời giải được mô tả theo nguyên tắc sau. Những phần liên quan đến xây dựng dữ liệu, huấn luyện mô hình và agent pipeline được giữ lại như lõi học thuật của đề tài. Những phần liên quan đến Django, FastAPI, Celery, dashboard và dữ liệu nhiều tầng được bổ sung như lớp hiện thực hóa hệ thống. Riêng các artifact thử nghiệm cũ như giao diện Streamlit được xem là prototype, có thể nhắc đến như bước phát triển trung gian nhưng không dùng làm kiến trúc đích cuối cùng.",
            0,
        ),
        ("3.2 Quy trình xây dựng dữ liệu", 2),
        ("3.2.1 Thu thập mã nguồn do con người viết", 3),
        (
            "Bộ dữ liệu huấn luyện được xây dựng từ các repository C++ công khai với mục tiêu tách rõ hai nhóm phong cách: mã cơ bản và mã hướng đối tượng. Tiêu chí lọc cần ưu tiên hai điểm: tính đại diện cho bài toán sinh viên và tính sạch khỏi ảnh hưởng của các mô hình LLM phổ biến. Vì vậy, dữ liệu đầu vào nên được ràng buộc theo mốc thời gian trước khi ChatGPT và các công cụ tương tự trở nên phổ biến rộng rãi.",
            0,
        ),
        ("3.2.2 Sinh dữ liệu đối ngẫu bằng mô hình AI", 3),
        (
            "Sau khi có tập human-written, mỗi file được dùng làm nguồn để sinh một phiên bản AI-generated giữ nguyên logic nhưng thay đổi phong cách trình bày, cách đặt tên, mức độ chú thích và mức độ chuẩn hóa. Cách làm này giúp tạo ra cặp đối chiếu gần nhau về nội dung thuật toán nhưng khác nhau về dấu vết hình thái, từ đó phù hợp với mục tiêu huấn luyện bộ phát hiện.",
            0,
        ),
        ("3.2.3 Chuẩn hóa, nhãn hóa và chuẩn bị cho pipeline dữ liệu", 3),
        (
            "Ở mức nghiên cứu, dữ liệu được lưu dưới dạng JSONL để thuận tiện cho huấn luyện. Ở mức hệ thống, dữ liệu thô và metadata được ánh xạ sang lớp Bronze, đặc trưng học máy và token đầu vào được ánh xạ sang lớp Silver, còn kết quả phân tích và lời giải thích được ánh xạ sang lớp Gold. Cách ánh xạ này giúp cùng một bộ dữ liệu phục vụ được cả huấn luyện lẫn vận hành.",
            0,
        ),
        ("3.3 Phương pháp huấn luyện mô hình phân loại", 2),
        ("3.3.1 GraphCodeBERT làm bộ phân loại lõi", 3),
        (
            "Bộ phân loại lõi được xây dựng bằng cách fine-tune GraphCodeBERT cho tác vụ phân loại nhị phân AI/HUMAN. Việc chọn GraphCodeBERT thay vì một encoder ngôn ngữ tổng quát giúp tận dụng tốt hơn cấu trúc ngữ nghĩa của mã nguồn, đặc biệt trong các trường hợp file có class, kế thừa, phạm vi truy cập và logic phân tách theo nhiều hàm.",
            0,
        ),
        ("3.3.2 Sliding window và ensemble nhiều fold", 3),
        (
            "Do nhiều file C++ dài hơn giới hạn 512 token, đầu vào được chia theo sliding window có overlap để tránh mất ngữ cảnh. Ở giai đoạn huấn luyện, chiến lược nhiều fold giúp giảm rủi ro lệ thuộc vào một cách chia dữ liệu duy nhất. Khi suy luận, kết quả có thể được tổng hợp từ nhiều mô hình con hoặc từ một runtime nhẹ hơn tùy môi trường triển khai.",
            0,
        ),
        ("3.4 Pipeline suy luận theo agent workflow", 2),
        ("3.4.1 Router node", 3),
        (
            "Router node quyết định nên dùng nhánh mô hình nào cho file đầu vào, chủ yếu theo phân loại Basic hay OOP. Về lý thuyết, đây là bước định tuyến mô hình để tránh dùng một bộ phân loại duy nhất cho mọi kiểu mã. Về thực thi, Router có thể kết hợp LLM-based classification với heuristic fallback để giảm lỗi khi tín hiệu định tuyến mơ hồ.",
            0,
        ),
        ("3.4.2 Analyzer node", 3),
        (
            "Analyzer node là nơi tiêu tốn tài nguyên nhiều nhất. Thành phần này phụ trách gọi bộ phân loại học sâu, tổng hợp điểm trên các chunk mã nguồn và xây dựng đầu ra giải thích ở cấp token khi runtime hỗ trợ. Trong repo hiện tại, FastAPI service không trực tiếp gánh phần inference nặng mà ủy quyền cho GPU worker thông qua API, nhờ đó giữ cho lớp điều phối nhẹ và dễ triển khai hơn.",
            0,
        ),
        ("3.4.3 Judge node", 3),
        (
            "Judge node có nhiệm vụ rà soát độ tin cậy của kết quả. Nếu điểm số rơi vào vùng nhập nhằng hoặc xung đột với tín hiệu bổ trợ như perplexity, node này có thể kích hoạt một lần suy luận lại với nhánh mô hình thay thế. Cơ chế này không biến hệ thống thành một tác nhân tự trị hoàn chỉnh, nhưng đủ để tạo nên hành vi self-correction ở mức workflow.",
            0,
        ),
        ("3.4.4 Critique node và giải thích đa mức", 3),
        (
            "Critique node tổng hợp và diễn giải kết quả cho người dùng. Ở phiên bản nghiên cứu, phần giải thích dựa mạnh vào LIG heatmap và lời nhận xét từ LLM. Ở phiên bản hệ thống hiện tại, repo còn bổ sung thêm một lớp phân tích fingerprint dùng LightGBM và SHAP để diễn giải các đặc trưng stylometry. Do đó, giải thích trong đề tài nên được mô tả như một kiến trúc đa mức gồm giải thích token-level và feature-level, thay vì chỉ giới hạn ở LIG.",
            0,
        ),
        ("3.5 Kiến trúc ứng dụng và triển khai hệ thống", 2),
        ("3.5.1 Lớp giao diện và tiếp nhận yêu cầu", 3),
        (
            "Lớp giao diện hiện tại được triển khai bằng Django. Người dùng đăng nhập, nộp mã nguồn, theo dõi lịch sử và xem lại kết quả phân tích. Cách tiếp cận này thay thế cho giao diện Streamlit ban đầu vì Django phù hợp hơn với yêu cầu quản lý phiên làm việc, quản trị người dùng và mở rộng dashboard giám sát.",
            0,
        ),
        ("3.5.2 Lớp AI service và luồng SSE", 3),
        (
            "FastAPI đóng vai trò lớp điều phối AI. Service này nhận mã nguồn đã mã hóa Base64, kích hoạt agent workflow và trả về tiến trình bằng Server-Sent Events. Luồng SSE cho phép giao diện web hiển thị trạng thái phân tích theo thời gian thực mà không cần polling nặng, đồng thời phù hợp với tính chất nhiều bước của pipeline.",
            0,
        ),
        ("3.5.3 Tầng dữ liệu Bronze, Silver, Gold", 3),
        (
            "Một thay đổi quan trọng của đề tài là đưa kiến trúc dữ liệu Medallion vào hệ thống. Bronze lưu mã nguồn thô và metadata nộp bài. Silver lưu đặc trưng đã trích xuất và token phục vụ huấn luyện hoặc tái sử dụng. Gold lưu kết quả suy luận cuối cùng, giải thích, thời gian chạy và các chỉ số cần cho dashboard. Cách phân tầng này tạo nền cho kiểm toán dữ liệu, tái huấn luyện và theo dõi vận hành.",
            0,
        ),
        ("3.5.4 Tác vụ nền và khả năng mở rộng", 3),
        (
            "Celery được dùng để tách các công việc không cần phản hồi tức thì ra khỏi request-response chính, ví dụ đồng bộ dữ liệu thô sang object storage hoặc chuẩn bị đặc trưng cho các bước sau. Việc tách tác vụ nền giúp giảm độ trễ ở lớp người dùng và cho phép hệ thống mở rộng dần theo hướng MLOps.",
            0,
        ),
        ("3.5.5 GPU runtime linh hoạt", 3),
        (
            "Repo hiện hỗ trợ nhiều biến thể runtime như Colab worker, local GPU server và các module phục vụ offload. Điều này phản ánh ràng buộc thực tế của đề tài: phần mô hình nặng đòi hỏi VRAM lớn, trong khi máy phát triển thông thường không phải lúc nào cũng đủ tài nguyên. Bởi vậy, kiến trúc hybrid local-cloud là một lựa chọn triển khai mang tính thực dụng thay vì chỉ là chi tiết hạ tầng.",
            0,
        ),
        ("3.6 Ánh xạ giải pháp vào cấu trúc repo", 2),
        (
            "Về mặt tổ chức mã nguồn, phần giao diện web nằm trong src/django_web, lớp điều phối AI nằm trong src/fastapi_service, phần xử lý GPU tách trong src/local_gpu_server hoặc runtime tương đương, tác vụ nền nằm trong src/celery_workers, pipeline dữ liệu nằm trong src/data_pipeline, còn hạ tầng dùng chung nằm trong src/shared. Cách chia này làm rõ ranh giới trách nhiệm giữa các thành phần và là cơ sở tốt để trình bày kiến trúc ở mức triển khai.",
            0,
        ),
        ("3.7 Tổng kết chương", 2),
        (
            "Giải pháp đề xuất của đề tài nên được hiểu như sự kết hợp giữa một lõi học thuật và một lớp hệ thống. Lõi học thuật gồm dữ liệu C++, GraphCodeBERT, suy luận nhiều bước và giải thích mô hình. Lớp hệ thống gồm web app, AI service, dữ liệu nhiều tầng, tác vụ nền và runtime GPU linh hoạt. Cách nhìn này vừa trung thực với tiến trình phát triển của đề tài, vừa giúp báo cáo phản ánh đúng những gì repo hiện đang thể hiện.",
            0,
        ),
    ]

    tables = [
        TableSpec(
            insert_after="3.1.2 Phạm vi chính thức được sử dụng trong báo cáo cập nhật",
            title="Bảng 3.1. So sánh prototype nghiên cứu và hệ thống hiện tại",
            headers=["Khía cạnh", "Prototype nghiên cứu", "Hệ thống hiện tại"],
            widths=[1.4, 2.45, 2.55],
            rows=[
                ["Giao diện", "Streamlit phục vụ demo phân tích.", "Django phục vụ người dùng, lịch sử và dashboard."],
                ["AI orchestration", "FastAPI tập trung quanh luồng Colab.", "FastAPI làm lớp điều phối độc lập, gọi GPU runtime qua API."],
                ["Giải thích", "LIG heatmap và critique bằng LLM.", "LIG khi có GPU và thêm fingerprint SHAP cho đặc trưng."],
                ["Dữ liệu", "JSONL phục vụ nghiên cứu.", "Bronze/Silver/Gold phục vụ cả vận hành và MLOps."],
            ],
        ),
        TableSpec(
            insert_after="3.6 Ánh xạ giải pháp vào cấu trúc repo",
            title="Bảng 3.2. Ánh xạ thành phần giải pháp vào mã nguồn hiện tại",
            headers=["Thành phần", "Vai trò", "Vị trí trong repo"],
            widths=[1.5, 2.2, 2.7],
            rows=[
                ["Web application", "Tiếp nhận code, quản lý người dùng, hiển thị kết quả.", "src/django_web/"],
                ["AI service", "Điều phối pipeline phân tích và SSE.", "src/fastapi_service/"],
                ["GPU runtime", "Thực thi mô hình nặng và trả kết quả chunk.", "src/local_gpu_server/ và các runtime liên quan"],
                ["Background tasks", "Đẩy dữ liệu và xử lý tác vụ bất đồng bộ.", "src/celery_workers/"],
                ["Data pipeline", "Trích đặc trưng, token hóa, chuẩn bị Gold layer.", "src/data_pipeline/"],
                ["Shared infrastructure", "Config, DB, broker, logger, hashing.", "src/shared/"],
            ],
        ),
        TableSpec(
            insert_after="3.5.5 GPU runtime linh hoạt",
            title="Bảng 3.3. Vai trò của ba tầng dữ liệu trong hệ thống",
            headers=["Tầng", "Nội dung lưu trữ", "Mục đích sử dụng"],
            widths=[1.0, 2.7, 2.7],
            rows=[
                ["Bronze", "Mã nguồn thô, metadata nộp bài, hash và nguồn dữ liệu.", "Lưu vết đầu vào, đồng bộ dữ liệu thô và phục vụ audit."],
                ["Silver", "Đặc trưng ML, token DL và dữ liệu trung gian đã chuẩn hóa.", "Phục vụ huấn luyện, phân tích lại và tái sử dụng pipeline."],
                ["Gold", "Kết quả dự đoán, giải thích, điểm số và thống kê vận hành.", "Hiển thị kết quả, báo cáo dashboard và phân tích hậu kiểm."],
            ],
        ),
    ]
    return "Chương 3 cập nhật", "Giải pháp đề xuất", parts, tables


def build_document(file_name: str, chapter_title: str, subtitle: str, parts, tables) -> Path:
    doc = Document()
    set_page_layout(doc)
    ensure_styles(doc)
    add_title_page(doc, chapter_title, subtitle)

    remaining_tables = list(tables)
    for text, level in parts:
        if level == 0:
            add_paragraph(doc, text)
        else:
            add_heading(doc, text, level)
            to_insert = [table for table in remaining_tables if table.insert_after == text]
            for table in to_insert:
                add_table(doc, table)
                remaining_tables.remove(table)

    # Add any remaining tables at the end of the nearest section.
    for table in remaining_tables:
        add_table(doc, table)

    out_path = OUT_DIR / file_name
    doc.save(out_path)
    return out_path


def write_changelog(ch2: Path, ch3: Path) -> Path:
    changelog = ROOT / "report_updates" / "CHAPTER_CHANGELOG.md"
    changelog.write_text(
        "\n".join(
            [
                "# Changelog cập nhật chương luận văn",
                "",
                "## Tệp được tạo",
                f"- `{ch2.name}`",
                f"- `{ch3.name}`",
                "",
                "## Chương 2",
                "- Viết lại phần mở đầu để bám đúng mục tiêu hiện tại của đề tài.",
                "- Giữ lõi lý thuyết về Transformer, CodeBERT, GraphCodeBERT và XAI.",
                "- Bổ sung cầu nối học thuật giữa code authorship, stylometry và AI code detection.",
                "- Mở rộng phần lý thuyết để bao phủ suy luận đa tín hiệu và kiến trúc dữ liệu nhiều tầng.",
                "- Tái cấu trúc research gaps theo bốn khoảng trống bám sát hiện trạng repo.",
                "",
                "## Chương 3",
                "- Viết lại định hướng chương theo hai lớp: prototype nghiên cứu và hệ thống hiện tại.",
                "- Chuyển trọng tâm giao diện từ Streamlit sang Django như kiến trúc chính thức.",
                "- Mô tả FastAPI là lớp điều phối AI và SSE thay cho mô tả backend đơn khối.",
                "- Bổ sung Medallion architecture, Celery và dữ liệu Bronze/Silver/Gold.",
                "- Điều chỉnh phần giải thích từ một hướng LIG duy nhất sang giải thích đa mức gồm LIG và SHAP fingerprint.",
                "- Loại bỏ việc mô tả mutation test như thành phần chắc chắn đang chạy trong pipeline chính.",
                "",
                "## Ghi chú",
                "- Hai file Word là bản viết mới, không ghi đè lên `Chuong 2+3.docx` gốc.",
                "- Chưa có xác nhận visual QA bằng render PNG vì môi trường hiện tại thiếu LibreOffice/soffice.",
            ]
        ),
        encoding="utf-8",
    )
    return changelog


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ch2_title, ch2_subtitle, ch2_parts, ch2_tables = chapter_2_content()
    ch3_title, ch3_subtitle, ch3_parts, ch3_tables = chapter_3_content()

    ch2 = build_document("Chuong_2_cap_nhat.docx", ch2_title, ch2_subtitle, ch2_parts, ch2_tables)
    ch3 = build_document("Chuong_3_cap_nhat.docx", ch3_title, ch3_subtitle, ch3_parts, ch3_tables)
    write_changelog(ch2, ch3)


if __name__ == "__main__":
    main()
