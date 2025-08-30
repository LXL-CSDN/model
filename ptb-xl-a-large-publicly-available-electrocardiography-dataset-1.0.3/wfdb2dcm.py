# wfdb2dcm_correct_official.py
"""
Correct version based on pydicom official documentation
"""
import sys, datetime
import numpy as np
import wfdb
import pydicom
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

def wfdb_to_dicom_correct_official(record_name: str, wfdb_dir: str, out_path: str):
    # --- 1. Read WFDB ---
    rec_path = os.path.join(wfdb_dir, record_name)
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal.astype("float32")
    fs = rec.fs
    leads = rec.sig_name
    n_samp, n_chan = sig.shape

    print(f"📊 Original Data Info:")
    print(f"   Shape: {sig.shape}")
    print(f"   Sampling Rate: {fs} Hz")
    print(f"   Leads: {leads}")
    print(f"   Data Range: [{sig.min():.3f}, {sig.max():.3f}] mV")

    # --- 2. Data Conversion ---
    sig_uv = sig * 1000.0  # mV -> µV
    sig_uv = np.clip(sig_uv, -32767, 32767)
    sig_int16 = np.round(sig_uv).astype(np.int16)
    
    print(f"   Converted Range: [{sig_int16.min()}, {sig_int16.max()}] (int16)")

    # --- 3. DICOM File Meta Information ---
    sop_instance_uid = pydicom.uid.generate_uid()
    
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1000"

    # --- 4. Main Dataset ---
    ds = FileDataset(out_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # SOP Tags
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.SOPInstanceUID = sop_instance_uid
    
    # Patient/Study Info
    ds.PatientName = "Anon^PTBXL"
    ds.PatientID = "PTBXL"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    
    ds.Modality = "ECG"
    ds.Manufacturer = "PTBXL-Converter"

    # --- 5. Waveform Sequence - Based on official documentation ---
    wv_item = Dataset()
    
    # Basic Waveform Info
    wv_item.WaveformOriginality = "ORIGINAL"
    wv_item.NumberOfWaveformChannels = n_chan
    wv_item.NumberOfWaveformSamples = n_samp
    wv_item.SamplingFrequency = float(fs)
    
    # Time related parameters
    wv_item.MultiplexGroupTimeOffset = 0.0
    wv_item.TriggerTimeOffset = 0.0
    
    # Complete Waveform bit definitions
    wv_item.WaveformBitsAllocated = 16
    wv_item.WaveformSampleInterpretation = "SS"
    
    # Interleave data and convert to bytes
    # Reshape data to (n_samp, n_chan) then convert to bytes
    arr = sig_int16.reshape(n_samp, n_chan)
    wf_bytes = arr.tobytes()
    wv_item.WaveformData = wf_bytes

    # --- 6. Channel Definition Sequence - Based on official documentation format ---
    ch_seq = []
    
    # Standard lead code mapping
    lead_codes = {
        'I': ('5.6.3-9-1', 'Lead I (Einthoven)'),
        'II': ('5.6.3-9-2', 'Lead II'),
        'III': ('5.6.3-9-61', 'Lead III'),
        'AVR': ('5.6.3-9-62', 'Lead aVR'),
        'AVL': ('5.6.3-9-63', 'Lead aVL'),
        'AVF': ('5.6.3-9-64', 'Lead aVF'),
        'V1': ('5.6.3-9-3', 'Lead V1'),
        'V2': ('5.6.3-9-4', 'Lead V2'),
        'V3': ('5.6.3-9-5', 'Lead V3'),
        'V4': ('5.6.3-9-6', 'Lead V4'),
        'V5': ('5.6.3-9-7', 'Lead V5'),
        'V6': ('5.6.3-9-8', 'Lead V6')
    }
    
    for ch_idx, lead in enumerate(leads):
        chdef = Dataset()
        
        # According to official documentation, set WaveformBitsStored in channel definition
        chdef.ChannelSampleSkew = 0.0
        chdef.WaveformBitsStored = 16
        
        # Lead code sequence
        chdef.ChannelSourceSequence = [Dataset()]
        source = chdef.ChannelSourceSequence[0]
        
        std_lead = lead.upper()
        if std_lead in lead_codes:
            source.CodeValue = lead_codes[std_lead][0]
            source.CodingSchemeDesignator = "SCPECG"
            source.CodingSchemeVersion = "1.3"
            source.CodeMeaning = lead_codes[std_lead][1]
        else:
            source.CodeValue = "1.0"
            source.CodingSchemeDesignator = "PYDICOM"
            source.CodingSchemeVersion = "1.0"
            source.CodeMeaning = lead
        
        # Calibration info (Type 1C - Optional)
        chdef.ChannelSensitivity = 1.0
        chdef.ChannelSensitivityUnitsSequence = [Dataset()]
        units_item = chdef.ChannelSensitivityUnitsSequence[0]
        units_item.CodeValue = "uV"
        units_item.CodingSchemeDesignator = "UCUM"
        units_item.CodeMeaning = "microvolt"
        
        ch_seq.append(chdef)
    
    wv_item.ChannelDefinitionSequence = Sequence(ch_seq)
    ds.WaveformSequence = Sequence([wv_item])

    # --- 7. Save and Verify ---
    try:
        ds.save_as(out_path, write_like_original=False)
        
        # Verify file
        verify_ds = pydicom.dcmread(out_path)
        ws = verify_ds.WaveformSequence[0]
        
        print(f"DICOM ECG saved successfully: {out_path}")
        print(f"   WaveformBitsAllocated: {ws.WaveformBitsAllocated}")
        print(f"   WaveformSampleInterpretation: {ws.WaveformSampleInterpretation}")
        
        # Check WaveformBitsStored in channel definition
        ch = ws.ChannelDefinitionSequence[0]
        print(f"   Channel WaveformBitsStored: {ch.WaveformBitsStored}")
        
        # Verify data
        expected_size = n_samp * n_chan * 2
        actual_size = len(ws.WaveformData)
        print(f"   Data size: {actual_size} bytes (Expected: {expected_size})")
        
        if actual_size == expected_size:
            print("   Data integrity verification passed")
        
        # Use pydicom's official method to verify data decoding
        try:
            decoded_arr = verify_ds.waveform_array(0)
            print(f"   Data decoding successful: {decoded_arr.shape}")
            print(f"   Decoded range: [{decoded_arr.min():.1f}, {decoded_arr.max():.1f}]")
        except Exception as decode_error:
            print(f"   Data decoding failed: {decode_error}")
        
        return True
        
    except Exception as e:
        print(f"Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python wfdb2dcm_correct_official.py <record_name> <wfdb_dir> <output.dcm>")
        sys.exit(1)
    
    rec_base, wfdb_dir, out_file = sys.argv[1:]
    
    print(f"Official Documentation Standard Version")
    print(f"   Converting: {rec_base} -> {out_file}")
    
    success = wfdb_to_dicom_correct_official(rec_base, wfdb_dir, out_file)
    
    if success:
        print(f"\n Converted successfully according to pydicom official documentation!")
        print(f"   Key Fixes:")
        print(f"   • Set WaveformBitsStored=16 in ChannelDefinitionSequence")
        print(f"   • Used correct data interleaving format")
        print(f"   • Followed the complete structure from official documentation")
        print(f"\n Test:")
        print(f"   1. Verify if ECG viewer still reports 'bits stored definition' error")
        print(f"   2. Check if waveform displays correctly")
    else:
        print(f"\n Conversion failed")