import numpy as np
import h5py
import pycbc.noise
import pycbc.psd
import pycbc.detector
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigmasq

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
PARAMS = {
    'seed': 42,                  # シード
    'num_samples': 100,          # サンプル数
    'output_file': 'train_data.hdf5',
    
    'mass_range': [30, 80],
    'spin_range': [0, 0.998],
    'snr_range':  [10, 50],
    
    'duration': 32.0,
    'sample_rate': 2048,
    'f_lower': 20.0,
    'approximant': 'IMRPhenomPv2',
    
    'slice_start': -0.20,
    'slice_end':   +0.05
}

# 感度ファイル
# H1とL1で同じファイルを指定します
PSD_FILES = {
    'H1': 'sensitivities/aligo_O4high.txt',  # 同じファイル
    'L1': 'sensitivities/aligo_O4high.txt',  # 同じファイル
    'V1': 'sensitivities/avirgo_O4high_NEW.txt', # Virgo用 (別途DL推奨)
    'K1': 'sensitivities/kagra_128Mpc.txt'   # 先ほどDLしたもの
}

# ==========================================
# 2. 関数定義
# ==========================================

def sample_parameters():
    # Numpyのシードが固定されていれば、ここの結果も固定されます
    m1 = np.random.uniform(*PARAMS['mass_range'])
    m2 = np.random.uniform(*PARAMS['mass_range'])
    mass1 = max(m1, m2)
    mass2 = min(m1, m2)

    spin1z = np.random.uniform(*PARAMS['spin_range'])
    spin2z = np.random.uniform(*PARAMS['spin_range'])

    ra = np.random.uniform(0, 2 * np.pi)
    dec = np.arcsin(np.random.uniform(-1, 1))
    pol = np.random.uniform(0, 2 * np.pi)
    inc = np.arccos(np.random.uniform(-1, 1))
    coa_phase = np.random.uniform(0, 2 * np.pi)

    target_snr = np.random.uniform(*PARAMS['snr_range'])

    return {
        'mass1': mass1, 'mass2': mass2,
        'spin1z': spin1z, 'spin2z': spin2z,
        'ra': ra, 'dec': dec,
        'inclination': inc, 'polarization': pol,
        'coa_phase': coa_phase,
        'target_snr': target_snr
    }

def load_psd(det_name, flen, delta_f):
    path = PSD_FILES.get(det_name)
    try:
        psd = pycbc.psd.from_txt(path, flen, delta_f, PARAMS['f_lower'], is_asd_file=True)
    except:
        if det_name == 'K1':
            psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, PARAMS['f_lower'])
            psd.data *= 10.0
        else:
            psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, PARAMS['f_lower'])
    return psd

# ==========================================
# 3. メイン生成ループ
# ==========================================
def generate_dataset():
    # Numpy全体のシードを固定 (パラメータ生成用)
    np.random.seed(PARAMS['seed'])
    
    num_samples = PARAMS['num_samples']
    sample_rate = PARAMS['sample_rate']
    slice_len_seconds = PARAMS['slice_end'] - PARAMS['slice_start']
    slice_points = int(slice_len_seconds * sample_rate)

    print(f"Generating {num_samples} samples (Seed: {PARAMS['seed']})...")

    data_buffer = {
        'h1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'l1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'v1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'k1_strain': np.zeros((num_samples, slice_points), dtype='float32')
    }
    
    param_buffer = {
        'mass1': np.zeros(num_samples),
        'mass2': np.zeros(num_samples),
        'spin1z': np.zeros(num_samples),
        'spin2z': np.zeros(num_samples),
        'ra': np.zeros(num_samples),
        'dec': np.zeros(num_samples),
        'injection_snr': np.zeros(num_samples)
    }

    detectors = {name: pycbc.detector.Detector(name) for name in ['H1', 'L1', 'V1', 'K1']}
    
    # 検出器ごとにユニークなオフセットを持たせるためのインデックス
    det_indices = {'H1': 0, 'L1': 1, 'V1': 2, 'K1': 3}

    for i in range(num_samples):
        p = sample_parameters()
        ref_distance = 1000.0
        tc_geocent = PARAMS['duration'] / 2.0

        hp, hc = get_td_waveform(
            approximant=PARAMS['approximant'],
            mass1=p['mass1'], mass2=p['mass2'],
            spin1z=p['spin1z'], spin2z=p['spin2z'],
            distance=ref_distance,
            inclination=p['inclination'],
            coa_phase=p['coa_phase'],
            delta_t=1.0/sample_rate,
            f_lower=PARAMS['f_lower']
        )
        hp.start_time = tc_geocent + hp.start_time
        hc.start_time = tc_geocent + hc.start_time

        net_snr_sq = 0.0
        signals_no_noise = {}

        for name, det in detectors.items():
            fp, fc = det.antenna_pattern(p['ra'], p['dec'], p['polarization'], tc_geocent)
            dt = det.time_delay_from_earth_center(p['ra'], p['dec'], tc_geocent)
            
            sig = fp * hp + fc * hc
            sig_shifted = sig.cyclic_time_shift(dt)
            sig_shifted.resize(int(PARAMS['duration'] * sample_rate))
            
            flen = int(sig_shifted.duration * sig_shifted.sample_rate / 2) + 1
            psd = load_psd(name, flen, sig_shifted.delta_f)
            
            snr_sq = sigmasq(sig_shifted, psd=psd, low_frequency_cutoff=PARAMS['f_lower'])
            net_snr_sq += snr_sq
            signals_no_noise[name] = (sig_shifted, psd)

        current_net_snr = np.sqrt(net_snr_sq)
        if current_net_snr == 0: current_net_snr = 1e-10
        scaling_factor = current_net_snr / p['target_snr']

        for name, (sig, psd) in signals_no_noise.items():
            sig_final = sig / scaling_factor
            
            # ノイズ生成用のシード計算
            # サンプル番号と検出器IDを組み合わせてユニークかつ再現可能なシードを作る
            # seed = Base + (sample_idx * 100) + det_idx
            noise_seed = PARAMS['seed'] + (i * 100) + det_indices[name]
            
            # seedを指定してノイズ生成
            noise = pycbc.noise.noise_from_psd(len(sig_final), sig_final.delta_t, psd, seed=noise_seed)
            noise.start_time = sig_final.start_time
            strain = sig_final + noise

            tc_idx = int(tc_geocent * sample_rate)
            idx_start = tc_idx + int(PARAMS['slice_start'] * sample_rate)
            idx_end   = tc_idx + int(PARAMS['slice_end']   * sample_rate)
            
            data_slice = strain.numpy()[idx_start:idx_end]
            if len(data_slice) < slice_points:
                data_slice = np.pad(data_slice, (0, slice_points - len(data_slice)))
            elif len(data_slice) > slice_points:
                data_slice = data_slice[:slice_points]
            
            key_name = name.lower() + '_strain'
            data_buffer[key_name][i] = data_slice

        param_buffer['mass1'][i] = p['mass1']
        param_buffer['mass2'][i] = p['mass2']
        param_buffer['spin1z'][i] = p['spin1z']
        param_buffer['spin2z'][i] = p['spin2z']
        param_buffer['ra'][i] = p['ra']
        param_buffer['dec'][i] = p['dec']
        param_buffer['injection_snr'][i] = p['target_snr']

        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples} samples.")

    print("Saving to HDF5 (ggwd structure)...")
    with h5py.File(PARAMS['output_file'], 'w') as f:
        grp_samples = f.create_group('injection_samples')
        grp_params = f.create_group('injection_parameters')
        
        for key, val in data_buffer.items():
            grp_samples.create_dataset(key, data=val)
            
        for key, val in param_buffer.items():
            grp_params.create_dataset(key, data=val)

    print(f"Done. Saved to {PARAMS['output_file']}")

if __name__ == "__main__":
    generate_dataset()