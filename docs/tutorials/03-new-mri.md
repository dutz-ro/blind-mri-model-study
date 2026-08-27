# Process a New MRI

Confirm that you are authorized to process the study and that the source folder is local.

Estimate space first:

```bash
./bin/mristudy estimate-space --input /absolute/path/to/dicom
```

```bash
./bin/mristudy init-study \
  --input /absolute/path/to/dicom \
  --profile knee-meniscus-v1

./bin/mristudy run --study <study-id> --through privacy-review
```

Review the anonymization HTML and pixel contact material locally. Reject the study if any identity
or unexplained annotation remains. After a satisfactory human review:

```bash
./bin/mristudy approve privacy-review --study <study-id> --note "Reviewed every series"
./bin/mristudy export --study <study-id>
```
