# Atlas IRM Private

Vizualizator privat, trilingv, pentru un raport IRM exploratoriu și non-diagnostic,
publicat exclusiv prin Tailscale.

## Limite

- Nu este dispozitiv medical, diagnostic sau sfat medical.
- Aplicația este urmărită în Git; datele studiului nu sunt.
- `report-bundle.json` și `evidence/` sunt ignorate intenționat: ele conțin
  rezultatele și secțiunile unui studiu concret.
- Nu adăuga fișiere DICOM, rapoarte generate sau jurnale în repository.

## Cum funcționează

Paginile nu conțin niciun text de studiu. Tot ce se vede provine din două surse
încărcate la runtime:

- `report-bundle.json` — rezultatele blocate ale studiului, generate cu
  `mristudy viewer-bundle --study <id>`;
- `i18n/{ro,en,fr}.json` — interfața și vocabularul clinic, în trei limbi.

Vocabularul clinic al schemei este enumerat, deci se traduce exact. Textul liber
produs de model rămâne verbatim, marcat vizibil ca text netradus: a-l traduce ar
însemna să inventăm formulări clinice pe care modelul nu le-a produs.

## Pagini

- `index.html` — constatările modelului, cu dovezile citate, galerie de imagini
  și panoul zonei vasculare.
- `report.html` — raport radiologic structurat, compus determinist din câmpurile
  studiului. Structurile pe care studiul nu le-a analizat sunt listate explicit
  ca neevaluate, pentru că tăcerea într-un raport radiologic se citește ca
  normalitate.
- `options.html` — subiecte educaționale de discutat cu un medic.

Limba se alege din bara de sus, se reține în `localStorage` și poate fi impusă
prin `?lang=ro|en|fr` pentru a partaja o legătură într-o limbă anume.

## Zona vasculară

Zona vasculară apare lângă fiecare constatare și într-un panou agregat. Peste
tot este însoțită de baza estimării, de încredere și de două mențiuni fixe:
vascularizația nu a fost observată direct, iar zona nu determină reparabilitatea
și nu susține o recomandare de tratament.

## Instalare locală

Aplicația folosește doar biblioteca standard Python și module ES native.

```bash
mristudy viewer-bundle --study <id>
cp -r index.html report.html options.html styles.css server.py js i18n /home/openclaw/atlas-irm/
cp <study>/reports/viewer/report-bundle.json /home/openclaw/atlas-irm/
cp -r <study>/reports/viewer/evidence /home/openclaw/atlas-irm/evidence
python3 server.py
```

Serverul ascultă exclusiv pe `127.0.0.1:18043` și trimite o politică de securitate
strictă: `default-src 'self'`, fără scripturi inline și fără resurse externe.

## Serviciu systemd

```bash
cp atlas-irm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now atlas-irm.service
```

## Acces Tailscale

```bash
tailscale serve --bg --https=10450 http://127.0.0.1:18043
```

Nu folosi `tailscale funnel` pentru această aplicație.

## Verificări rapide

```bash
python3 -m py_compile server.py
for f in js/*.js; do node --check "$f"; done
python3 - <<'PY'
import json, pathlib
base = None
for path in sorted(pathlib.Path("i18n").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = set()
    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                keys.add(prefix + key)
                walk(value, prefix + key + ".")
    walk(data)
    base = keys if base is None else base
    assert keys == base, f"{path} has a different key set"
print("dictionaries aligned")
PY
```
