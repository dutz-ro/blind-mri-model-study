# Cloud Cleanup

Create and inspect the deletion plan:

```bash
./bin/mristudy cloud destroy-plan --study <id>
./bin/mristudy approve cloud-destroy --study <id>
./bin/mristudy cloud destroy --study <id>
```

This release deletes only exact DICOM instances from the approved upload plan. Vertex endpoint,
deployed model, DICOM Store, Healthcare dataset, bucket, IAM, and billing cleanup must additionally
be verified by the operator if those resources were created for the study.
