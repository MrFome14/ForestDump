import pandas as pd
import joblib
import sys
import os
import argparse
from datetime import datetime


def run_scan(file_p, model_p, feat_p, mode):
    model = joblib.load(model_p)
    f_names = joblib.load(feat_p)

    if mode == 'all':
        rows = None
        print("Режим: Полное сканирование")
    else:
        rows = 200000
        print(f"Режим: Быстрое сканирование ({rows} строк)")

    df = pd.read_csv(file_p, comment='#', nrows=rows)
    df = df.dropna(subset=['Label'])

    # Считаем признаки
    df['Dur'] = df['Dur'].replace(0, 0.000001)
    df['Pkts_Sec'] = df['TotPkts'] / df['Dur']
    df['Bytes_Sec'] = df['TotBytes'] / df['Dur']
    df['Bytes_Pkt'] = df['TotBytes'] / df['TotPkts']

    df_enc = pd.get_dummies(df, columns=['Proto', 'State', 'Dir'])
    X = df_enc.reindex(columns=f_names, fill_value=0)

    df['is_bot'] = model.predict(X)

    # Берем только уникальные IP-адреса, которые модель пометила как ботов
    bad_hosts = df[df['is_bot'] == 1]['SrcAddr'].unique()

    print(f"\n--- Результаты: {os.path.basename(file_p)} ---")

    if len(bad_hosts) > 0:
        print(f"ОБНАРУЖЕНО ПОДОЗРИТЕЛЬНЫХ ХОСТОВ: {len(bad_hosts)}")
        print("\nСписок IP:")
        for ip in bad_hosts[:15]:  # Показываем первые 15 в консоли
            print(f" [!] {ip}")

        if not os.path.exists('detections'):
            os.makedirs('detections')

        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"detections/suspect_hosts_{t}.txt"

        # Сохраняем просто список адресов в столбик
        with open(out, 'w') as f:
            for ip in bad_hosts:
                f.write(f"{ip}\n")

        print(f"\nЧистый список адресов сохранен в: {out}")
    else:
        print("\nУгроз не обнаружено. Все хосты выглядят надежно.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Путь к binetflow файлу")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min", action="store_true")

    args = parser.parse_args()
    mode = 'all' if args.all else 'min'

    if not os.path.exists(args.file):
        print(f"Файл {args.file} не найден")
    else:
        # Убедись, что пути к твоим моделям совпадают
        run_scan(args.file, 'models/universal_botnet_detector.pkl', 'models/universal_features.pkl', mode)