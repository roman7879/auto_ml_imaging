"""Script to download TCIA CBIS-DDSM and store it in Healthcare API.

This script will download full mammo images from TCIA CBIS-DDSM dataset.
Specifically, it will retrieve images that correspond to BI-RADS breast density
"2" (scattered density) and breast density "3" (heterogeneously dense). It will
then upload the images to Cloud Healthcare API.

Dataset is found here:
https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

This guide requires a TCIA API key. Read the following guide to retrieve a TCIA
API key:
https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+%28REST+API%29+Usage+Guide

Alternatively, you can just use the following key:
0705ffc5-8148-4d7d-904c-915d661116af
TODO(b/73360870): remove this before OSS.

This script will use application default credentials when invoking Healthcare
API.

Example usage:
TCIA_API_KEY=...
PROJECT_ID=...
DATASET_ID=...
DICOM_STORE_ID=...

python
third_party/cloud/healthcare/imaging/ml_training/codelab/store_tcia_in_hc_api.py
--api_key=${TCIA_API_KEY} --project_id=${PROJECT_ID} --dataset_id=${DATASET_ID}
--dicom_store_id=${DICOM_STORE_ID}

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
from email import encoders
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import json
import logging
import multiprocessing
import signal
import StringIO
import sys
import httplib2
from oauth2client.client import GoogleCredentials

# Output this module's logs (INFO and above) to stdout.
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

FLAGS = None

_TCIA_BASE_PATH = "https://services.cancerimagingarchive.net/services/v3/TCIA/query"
_CREDENTIALS = GoogleCredentials.get_application_default()
# Labels files that contain breast density labels and UIDs.
_LABEL_PATHS = [
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_train_set.csv",
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_train_set.csv",
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_test_set.csv",
    "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_test_set.csv",
]
_BREAST_DENSITY_COLUMN = {"breast_density", "breast density"}
_IMAGE_FILE_PATH_COLUMN = {"image file path"}


def main():
  # Download UIDs for breast density 2 and 3.
  http = httplib2.Http(timeout=60)
  series_instance_uids = set()
  for path in _LABEL_PATHS:
    resp, content = http.request(path, method="GET")
    assert resp.status == 200, "Failed to download label files from: " + path
    r = csv.reader(StringIO.StringIO(content), delimiter=",")
    header = r.next()
    breast_density_column = -1
    image_file_path_column = -1
    for idx, h in enumerate(header):
      if h in _BREAST_DENSITY_COLUMN:
        breast_density_column = idx
      if h in _IMAGE_FILE_PATH_COLUMN:
        image_file_path_column = idx
    assert breast_density_column != -1, "breast_density column not found"
    assert image_file_path_column != -1, "image file path column not found"
    for row in r:
      density = row[breast_density_column]
      if density != "2" and density != "3":
        continue
      dicom_uids = row[image_file_path_column].split("/")
      study_instance_uid, series_instance_uid = dicom_uids[1], dicom_uids[2]
      if FLAGS.has_study_uid and FLAGS.has_study_uid != study_instance_uid:
        continue
      series_instance_uids.add(series_instance_uid)

  # Register KeyboardInterrupt handler
  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  pool = multiprocessing.Pool(processes=FLAGS.max_concurrency)
  signal.signal(signal.SIGINT, original_sigint_handler)

  total_instances_count = len(series_instance_uids)
  logger.info("There are %s instances to upload...", total_instances_count)
  try:
    for idx, _ in enumerate(
        pool.imap_unordered(_DownloadInstanceFromTCIA, series_instance_uids),
        1):
      # Provide an update every 100 instances uploaded.
      if idx % 100 == 0:
        logger.info("Imported [%s/%s] instances so far...", idx,
                    total_instances_count)
  except KeyboardInterrupt:
    logger.error("Received keyboard interrupt - quitting...")
    pool.terminate()
    pool.join()
    raise
  except Exception:
    pool.terminate()
    pool.join()
    raise
  finally:
    pool.close()
    pool.join()
  logger.info("Successfully uploaded all instances!")


def _DownloadInstanceFromTCIA(series_instance_uid):
  # type: (str) -> None
  """Downloads TCIA CBIS-DDSM an stores it in Healthcare API."""
  sop_instance_path = (
      "%s/getSOPInstanceUIDs?SeriesInstanceUID=%s&api_key=%s" %
      (_TCIA_BASE_PATH, series_instance_uid, FLAGS.tcia_api_key))
  http = httplib2.Http(timeout=60)  # default is 5 seconds.
  resp, content = http.request(sop_instance_path, method="GET")
  assert resp.status == 200, (
      "Failed getting instance UID for series " + series_instance_uid)
  decoder = json.JSONDecoder()
  json_out = decoder.decode(content)
  assert len(json_out) == 1, (
      "There should be only one instance UID for series: " +
      series_instance_uid)
  sop_instance_uid = json_out[0]["sop_instance_uid"]
  image_path = (
      "%s/getSingleImage?SeriesInstanceUID=%s&SOPInstanceUID=%s&api_key=%s" %
      (_TCIA_BASE_PATH, series_instance_uid, sop_instance_uid,
       FLAGS.tcia_api_key))
  resp, content = http.request(image_path, method="GET")
  assert resp.status == 200, (
      "Failed retrieving image with instance UID: " + sop_instance_uid)
  _UploadInstanceToHealthcareAPI(sop_instance_uid, content)


def _UploadInstanceToHealthcareAPI(sop_instance_uid, inst):
  # type: (str, str) -> None
  """Uploads instances in Healthcare API."""
  http = httplib2.Http(timeout=60)
  http = _CREDENTIALS.authorize(http)

  related = MIMEMultipart("related", boundary="boundary")
  setattr(related, "_write_headers", lambda self: None)

  mime_attach = MIMEApplication(inst, "dicom", _encoder=encoders.encode_noop)
  related.attach(mime_attach)

  multipart_boundary = related.get_boundary()
  headers = {}
  headers["content-type"] = (
      'multipart/related; type="application/dicom"; boundary="%s"'
  ) % multipart_boundary
  body = related.as_string()
  path = "https://healthcare.googleapis.com/v1alpha/projects/{}/locations/{}/datasets/{}/dicomStores/{}/dicomWeb/studies".format(
      FLAGS.project_id, FLAGS.location, FLAGS.dataset_id,
      FLAGS.dicom_store_id)
  resp, content = http.request(path, method="POST", headers=headers, body=body)
  if resp.status == 409:
    logger.warning("Instance with SOP Instance UID already exists: %s",
                   sop_instance_uid)
    return
  assert resp.status == 200, (
      "Failed to store instance %s, reason: %s, response: %s" %
      (sop_instance_uid, resp.reason, content))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--tcia_api_key", type=str, default="", help="TCIA API key.")

  parser.add_argument(
      "--project_id", type=str, default="", help="Cloud Healthcare project id.")
  parser.add_argument(
      "--location",
      type=str,
      default="us-central1",
      help="Location to store Cloud Healthcare dataset.")
  parser.add_argument(
      "--dataset_id", type=str, default="", help="Cloud Healthcare dataset id.")
  parser.add_argument(
      "--dicom_store_id",
      type=str,
      default="",
      help="Cloud Healthcare dicom store id.")
  parser.add_argument(
      "--has_study_uid",
      type=str,
      default=None,
      help=
      "If set, only instances having the specified Study UID will be uploaded.")

  parser.add_argument(
      "--max_concurrency", type=int, default=10, help="Max concurrency.")
  FLAGS, unparsed = parser.parse_known_args()
  main()
