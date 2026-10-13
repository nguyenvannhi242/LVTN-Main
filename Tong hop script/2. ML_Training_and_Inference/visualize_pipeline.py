import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
WORKSPACE = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
SAVE_PATH = os.path.join(WORKSPACE, "retrain ML models", "feature_selection_pipeline.png")

def draw_pipeline():
    # Set style
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
    ax.axis('off')
    
    # Define colors (Sleek dark/blue palette)
    bg_color = "#f4f6f9"
    box_color_input = "#2c3e50"     # Dark Slate
    box_color_s1 = "#1abc9c"        # Turquoise
    box_color_s2 = "#3498db"        # Blue
    box_color_s3 = "#9b59b6"        # Purple
    box_color_final = "#27ae60"     # Green
    text_light = "#ffffff"
    text_dark = "#2c3e50"
    arrow_color = "#7f8c8d"
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Coordinates of boxes
    # boxes format: (x, y, width, height, title, color, details)
    boxes = [
        # Raw Input Features
        (0.5, 7.5, 3.0, 1.0, "Raw Features (V8 Extractor)\n44 Đặc trưng C++", box_color_input, 
         "• Layout & Formatting (8)\n• Naming Conventions (5)\n• Structural Complexity (10)\n• Habits & Idioms (8)\n• Information Theory (3)\n• OOP Features (10)"),
        
        # Stage 1: Correlation Filter
        (4.5, 7.5, 3.5, 1.0, "STAGE 1: Correlation Filter\nBộ lọc đa cộng tuyến", box_color_s1,
         "• Ngưỡng lọc Pearson: |r| > 0.85\n• Drop 1 đặc trưng trong cặp có r cao\n• Giải quyết vấn đề đa cộng tuyến\n\n➔ Loại bỏ: 4 đặc trưng\n➔ Còn lại: 40 đặc trưng"),
        
        # Stage 2: Select K-Best
        (9.0, 7.5, 3.5, 1.0, "STAGE 2: SelectKBest (MI)\nLọc đặc trưng theo lượng thông tin", box_color_s2,
         "• Thuật toán: Mutual Information\n• Đo lường sự tương quan phi tuyến\n• Chọn tham số tối ưu: k = 33\n\n➔ Loại bỏ: 7 đặc trưng\n➔ Còn lại: 33 đặc trưng"),
        
        # Stage 3: Lasso L1 Regularization
        (9.0, 3.0, 3.5, 1.0, "STAGE 3: Lasso L1 Regularization\nBóp nghẹt trọng số hồi quy", box_color_s3,
         "• Bộ lọc: Logistic Regression L1\n• Penalty parameter: C = 0.5\n• Triệt tiêu các feature ít quan trọng\n  về mức hệ số bằng 0\n\n➔ Loại bỏ: 3 đặc trưng\n➔ Còn lại: 30 đặc trưng"),
         
        # Final Gold Features
        (4.5, 3.0, 3.5, 1.0, "FINAL FEATURES\n30 Đặc trưng Tinh túy", box_color_final,
         "• Đưa vào huấn luyện các mô hình chính:\n  - LightGBM (Winning Model)\n  - XGBoost\n  - Random Forest\n➔ Đạt Accuracy trên Test: 95.16%\n➔ Val F1-score: 94.23%")
    ]
    
    # Draw Boxes and text
    for x, y, w, h, title, col, details in boxes:
        # Draw main title box
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03", 
            facecolor=col, edgecolor="none", mutation_scale=0.2
        )
        ax.add_patch(rect)
        
        # Add Title text
        ax.text(
            x + w/2, y + h/2, title, color=text_light, fontsize=10, 
            weight='bold', ha='center', va='center', multialignment='center'
        )
        
        # Draw detail box underneath
        detail_rect = patches.FancyBboxPatch(
            (x, y - 1.6), w, 1.4, boxstyle="round,pad=0.03",
            facecolor="#ffffff", edgecolor="#bdc3c7", mutation_scale=0.2, linewidth=1.5
        )
        ax.add_patch(detail_rect)
        
        # Add Detail text
        ax.text(
            x + 0.15, y - 0.9, details, color=text_dark, fontsize=9.5,
            ha='left', va='center', multialignment='left', linespacing=1.3
        )
        
    # Draw Connecting Arrows
    # Arrow 1: Input -> Stage 1
    ax.annotate("", xy=(4.4, 8.0), xytext=(3.6, 8.0),
                arrowprops=dict(facecolor=arrow_color, edgecolor="none", width=4, headwidth=10))
    
    # Arrow 2: Stage 1 -> Stage 2
    ax.annotate("", xy=(8.9, 8.0), xytext=(8.1, 8.0),
                arrowprops=dict(facecolor=arrow_color, edgecolor="none", width=4, headwidth=10))
    
    # Arrow 3: Stage 2 -> Stage 3 (Vertical down arrow)
    ax.annotate("", xy=(10.75, 4.1), xytext=(10.75, 5.8),
                arrowprops=dict(facecolor=arrow_color, edgecolor="none", width=4, headwidth=10))
    
    # Arrow 4: Stage 3 -> Final Features
    ax.annotate("", xy=(8.1, 3.5), xytext=(8.9, 3.5),
                arrowprops=dict(facecolor=arrow_color, edgecolor="none", width=4, headwidth=10))
                
    # Add title header
    ax.text(
        7.0, 9.3, "QUY TRÌNH TUYỂN CHỌN ĐẶC TRƯNG 3 TẦNG (3-STAGE FEATURE SELECTION PIPELINE)", 
        color="#2c3e50", fontsize=14, weight='bold', ha='center', va='center'
    )
    
    ax.text(
        7.0, 9.0, "Áp dụng cấu hình k = 33 tối ưu - Huấn luyện mô hình phát hiện C++ AI Code", 
        color="#7f8c8d", fontsize=11, style='italic', ha='center', va='center'
    )
    
    plt.xlim(-0.5, 13.5)
    plt.ylim(0.5, 10.0)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Pipeline visualization saved to {SAVE_PATH}")

if __name__ == "__main__":
    draw_pipeline()
