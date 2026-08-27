from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def create_synthetic_mri(root: Path, slices: int = 12, burned_in_text: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    study_uid = generate_uid(entropy_srcs=["mristudy-demo", "study"])
    series_uid = generate_uid(entropy_srcs=["mristudy-demo", "series"])
    now = datetime.now()
    paths = []
    for index in range(slices):
        sop_uid = generate_uid(entropy_srcs=["mristudy-demo", "instance", str(index)])
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid(entropy_srcs=["mristudy-demo", "implementation"])
        path = root / f"slice-{index + 1:04d}.dcm"
        ds = FileDataset(path.name, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = generate_uid(entropy_srcs=["mristudy-demo", "frame"])
        ds.PatientName = "DEMO^PATIENT"
        ds.PatientID = "DEMO-MRN-001"
        ds.PatientBirthDate = "19800101"
        ds.InstitutionName = "Synthetic Hospital"
        ds.Modality = "MR"
        ds.SeriesDescription = "SAG 3D PD FS SYNTHETIC"
        ds.ProtocolName = "KNEE SYNTHETIC"
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.SeriesDate = now.strftime("%Y%m%d")
        ds.InstanceNumber = index + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(index)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [0.5, 0.5]
        ds.SliceThickness = 1.0
        ds.SpacingBetweenSlices = 1.0
        ds.RepetitionTime = 1200.0
        ds.EchoTime = 32.0
        ds.Laterality = "L"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M"]
        ds.Rows = 128
        ds.Columns = 128
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        y, x = np.ogrid[:128, :128]
        radius = 35 + index // 3
        array = np.zeros((128, 128), dtype=np.uint16)
        array[(x - 64) ** 2 + (y - 64) ** 2 <= radius**2] = 800 + index * 20
        array[(x - 53) ** 2 + (y - 68) ** 2 <= 10**2] = 1900
        if burned_in_text:
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            draw.text((2, 2), "DEMO PATIENT", fill=3000)
            array = np.asarray(image, dtype=np.uint16)
            ds.BurnedInAnnotation = "YES"
        else:
            ds.BurnedInAnnotation = "NO"
        ds.PixelData = array.tobytes()
        ds.save_as(path, enforce_file_format=True)
        path.chmod(0o600)
        paths.append(path)
    return {"study_uid": study_uid, "series_uid": series_uid, "files": len(paths)}
