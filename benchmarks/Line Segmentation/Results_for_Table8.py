import json

def print_table_row(protocol_name, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    m = data.get('final_metrics', {})
    
    # Format numbers for the table
    m_all = f"{m.get('mIoU(all)', 0):.4f}"
    m_m = f"{m.get('mIoU(m)', 0):.4f}"
    f1_50 = f"{m.get('F1@0.50', 0):.4f}"
    f1_75 = f"{m.get('F1@0.75', 0):.4f}"
    lca = f"{m.get('LCA (%)', 0):.2f}"
    rel_err = f"{m.get('RelCntErr', 0):.4f}"
    ssim = f"{m.get('MeanSSIM', 0):.4f}"

    print(f"--- {protocol_name} ---")
    print(f"LaTeX Row: {protocol_name} & {m_all} & {m_m} & {f1_50} & {f1_75} & {lca} & {rel_err} & {ssim} \\\\")
    print(f"Plain Text: mIoU(all)={m_all}, mIoU(m)={m_m}, F1@0.50={f1_50}, F1@0.75={f1_75}, LCA={lca}%, RelCntErr={rel_err}, MeanSSIM={ssim}\n")

if __name__ == "__main__":
    print_table_row("Protocol 1", "evaluation_protocol1.json")
    print_table_row("Protocol 2", "evaluation_protocol2.json")