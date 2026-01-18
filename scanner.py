import pandas as pd
import joblib
import sys
import os
import argparse
from datetime import datetime


def run_scan(file_p, model_p, feat_p, read_all, full_report):
    model = joblib.load(model_p)
    f_names = joblib.load(feat_p)

    rows = None if read_all else 200000
    print(f"Режим: {'Полное' if read_all else 'Быстрое'} сканирование")

    df = pd.read_csv(file_p, comment='#', nrows=rows)
    df = df.dropna(subset=['Label'])

    df['Dur'] = df['Dur'].replace(0, 0.000001)
    df['Pkts_Sec'] = df['TotPkts'] / df['Dur']
    df['Bytes_Sec'] = df['TotBytes'] / df['Dur']
    df['Bytes_Pkt'] = df['TotBytes'] / df['TotPkts']

    df_enc = pd.get_dummies(df, columns=['Proto', 'State', 'Dir'])
    X = df_enc.reindex(columns=f_names, fill_value=0)

    df['is_bot'] = model.predict(X)
    bots = df[df['is_bot'] == 1]

    print(f"\n--- Результаты: {os.path.basename(file_p)} ---")

    if not bots.empty:
        bad_ips = bots['SrcAddr'].unique()
        print(f"ПОДОЗРИТЕЛЬНЫХ ХОСТОВ: {len(bad_ips)}")

        if not os.path.exists('detections'):
            os.makedirs('detections')

        t = datetime.now().strftime("%Y%m%d_%H%M%S")

        if full_report:
            out = f"detections/full_report_{t}.csv"
            bots.to_csv(out, index=False)
            print(f"[!] Подробный отчет (CSV) сохранен: {out}")
        else:
            out = f"detections/ips_{t}.txt"
            with open(out, 'w') as f:
                f.write('\n'.join(bad_ips))
            print(f"[!] Список IP (TXT) сохранен: {out}")
            print("Первые 10 адресов:", bad_ips[:10])
    else:
        print("Угроз не обнаружено.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Путь к binetflow файлу")
    parser.add_argument("--all", action="store_true", help="Прочитать весь лог")
    parser.add_argument("--full", action="store_true", help="Выгрузить полный CSV отчет")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Файл {args.file} не найден")
    else:
        run_scan(args.file, 'models/universal_botnet_detector.pkl', 'models/universal_features.pkl', args.all,
                 args.full)