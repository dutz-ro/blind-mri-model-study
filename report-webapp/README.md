# Atlas IRM Private

Infrastructură și șabloane pentru prezentarea privată, prin Tailscale, a unui raport IRM exploratoriu și non-diagnostic.

## Limite

- Nu este dispozitiv medical, diagnostic sau sfat medical.
- Repository-ul conține numai infrastructura și șabloanele reutilizabile.
- Imaginile, rapoartele, fișierele DICOM, datele studiului și jurnalele nu trebuie adăugate în Git.
- Directorul local `evidence/` este ignorat intenționat.
- Fișierele locale `index.html`, `tratament.html` și `app.js` sunt ignorate deoarece conțin raportul și legăturile către dovezile unui studiu concret.

## Rulare locală

Aplicația folosește doar biblioteca standard Python. Pentru o instalare nouă, pornește de la fișierele neutre din `templates/`, creează local fișierele finale ignorate de Git și rulează:

```bash
python3 server.py
```

Serverul ascultă exclusiv pe `127.0.0.1:18043`.

## Dovezi vizuale

Viewer-ul local poate căuta imaginile aprobate în `evidence/`, folosind identificatori anonimi. Aceste fișiere și identificatorii concreți se furnizează separat, numai pe sistemul autorizat, și nu se comit.

Șablonul de raport prezintă modelele independent și permite desemnarea fiecărei imagini ca susținere, context, contra-dovadă sau proveniență neconfirmată. Marcajele și săgețile trebuie etichetate ca orientative dacă modelul nu a furnizat coordonate ori segmentări validate.

## Serviciu systemd

Fișierul `atlas-irm.service` presupune instalarea aplicației în:

```text
/home/openclaw/atlas-irm
```

După copierea locală a aplicației:

```bash
cp atlas-irm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now atlas-irm.service
```

## Acces Tailscale

Exemplu de expunere exclusiv în tailnet:

```bash
tailscale serve --bg --https=18043 http://127.0.0.1:18043
```

Nu folosi `tailscale funnel` pentru această aplicație.

## Verificări rapide

```bash
python3 -m py_compile server.py
node --check app.js
```
