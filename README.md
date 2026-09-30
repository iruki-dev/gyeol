# gyeol (결)

녹음된 노래를 **녹음 조건에 강건한 보컬 중간 표현(intermediate representation)**으로 변환하는 Python 엔진입니다.

연구 문서("A Minimal, Condition-Invariant Intermediate Representation of the Singing Voice for Korean-Speaking Singers")의 권고안인 **하이브리드 표현 (d)**를 구현합니다.

1. **해석 가능한 source–filter 코어**: 10 ms 프레임 단위의 피치, 음원(phonation), 필터(공명), 에너지 그룹
2. **한국어 음운 컨텍스트 토큰**: 평음·격음·경음 등 선행 자음의 후두 자질과 모음 시작 이후 경과 시간
3. **잔차 임베딩 z** (선택): 채널에 대해 적대적으로 학습된 32차원 잠재 벡터
4. **차원별 유효성 마스크** (필수): 별도의 nuisance side channel(SNR, 대역폭/코덱, 클리핑, AGC, 잔향, 분리 품질)로 결정됩니다.

> "녹음 조건과 상관없이 같은 사람에게 같은 값"은 **측정된 유효 영역 안에서만** 성립합니다.
> 그래서 gyeol은 유효 영역 밖의 값을 숫자로 내보내지 않고 **결측(invalid)**으로 표시하며, 그 이유를 함께 기록합니다.

## 설치

```bash
pip install -e .                 # numpy, scipy만 필요
pip install -e ".[audio]"        # soundfile (WAV/FLAC/MP3 로딩)
pip install -e ".[torch]"        # 잔차 임베딩 z 모델
pip install -e ".[separation]"   # HTDemucs 분리기 어댑터
pip install -e ".[dev]"          # pytest
```

## 빠른 시작

```python
import gyeol

rep = gyeol.analyze("take.wav", lyrics="사랑해 너를")   # 또는 analyze(array, sr)

rep.summary()                      # 차원별 중앙값·IQR·유효율·무효 사유
rep.tracks["h1h2c"].masked()       # 무효 프레임은 NaN
rep.tracks["cpps"].invalid_reasons # {"unvoiced": 0.2, "low_snr": 0.1, ...}
rep.nuisance                       # SNR, 대역폭, 코덱 의심, AGC, T60 ...
for note in rep.notes:             # 음표 단위 비브라토·지터·온셋·집계값
    print(note.syllable, note.features["vibrato_rate_hz"], note.valid["vibrato_rate_hz"])

rep.save("take.npz")
rep = gyeol.VocalRepresentation.load("take.npz")
```

CLI:

```bash
gyeol analyze take.wav -o take.npz --lyrics "사랑해" --summary
gyeol analyze take.wav --textgrid take.TextGrid        # MFA 강제 정렬 사용
gyeol spec                                             # 차원 명세표 출력 (--json)
gyeol fit-device reference.wav phone.wav -o phone.json # 기기 EQ 프로파일
gyeol analyze take.wav --device-profile phone.json
```

## 표현 구성

`gyeol spec`이 전체 명세(단위, 프레임율, 추정기, 유효 조건, 누락 정보, 검증 방법)를 출력합니다. 명세는 `gyeol.spec.DIMENSIONS`에 기계 판독 가능한 형태로 들어 있습니다.

| 그룹 | 차원 | 단위 / 프레임율 | 추정기 | 주요 유효 조건 |
|---|---|---|---|---|
| 피치 | `f0_hz`, `f0_cents`, `voicing_prob` | Hz, cents / 100 Hz | pYIN·YIN·SHS 합의(중앙값), 추적기 간 편차 → 신뢰도 | 다수결 유성, 신뢰도, 분리 품질 |
| 에너지 | `energy_rel_db` | 프레이즈 중앙값 대비 dB / 100 Hz | RMS | AGC 없음 (절대 SPL은 **의도적 누락**) |
| 음원 | `cpps` | dB / 100 Hz | Hillenbrand형, f0 추적 quefrency 탐색 | 프레임 SNR ≥ 30 dB, 코덱 없음, f0 상한 |
| 음원 | `band_aperiodicity`, `band_hnr` | dB × 5밴드 / 100 Hz | 하모닉 빗(comb) 잡음 밀도 추정 | SNR, 코덱, 분리 품질, 대역폭 ≥ 8 kHz |
| 음원 | `h1h2c`, `h2h4c`, `h1a1c`, `h1a3c` | dB / 100 Hz | 하모닉 진폭 + Iseli–Alwan 포먼트 보정 | F1·F2 분해 가능, \|F1−f0\|, \|F1−2f0\| > B1, 기기 HPF ≪ f0 |
| 음원 | `naq`, `qoq`, `rd` | 무차원 | IAIF 역필터링 | 안정 발성, 위상 보존 채널, f0 ≤ 500 Hz(잠정) |
| 음원 | `alpha_ratio`, `hammarberg`, `lh_ratio` | dB / 25 Hz | 대역 에너지 | 대역폭 ≥ 5 kHz, EQ 미적용 시 신뢰도 감소 |
| 음원 | `shr` | 비율 / 100 Hz | subharmonic-to-harmonic ratio | SNR, 코덱 |
| 필터 | `f1`–`f4`, `b1`–`b4` | Hz / 50 Hz | f0 < 350 Hz: STE-가중 LP, 이상: 비브라토 스윕 하모닉 풀링 | F_n ≥ 1.5·f0(잠정), 대역폭 |
| 필터 | `envelope` | 켑스트럼 24계수 / 50 Hz | True envelope (Röbel & Rodet) | 유성, 대역폭 |
| 필터 | `spr`, `r1_f0`, `r1_2f0`, `a1_p0` | dB, 비율 | SPR, 공명 튜닝, A1–P0 비음성 | F1 유효, P0 ≠ H1/H2, 모음 구간 |

음표 단위 차원 (`gyeol.spec.NOTE_DIMENSIONS`):
- 비브라토: 속도(rate), 폭(extent), 규칙성(CV). 2주기 이상일 때만 유효합니다.
- 지터·시머: 비브라토 < 15 cents인 지속음에서만, SNR ≥ 30 dB이고 샘플레이트 ≥ 19 kHz일 때 유효합니다.
- 온셋: 10–90% 상승 시간, f0 안정화 지연, 초기 비주기성, 초기 H1*–H2*
- 각 프레임 차원의 음표별 중앙값·IQR·유효율

선언된 누락(`gyeol.spec.OMISSIONS`): 절대 SPL, 들숨 소음, 공간감, 딕션 세부, 시각 정보, 미세 위상.

## 유효성 마스크와 nuisance side channel

모든 임계값은 `ValidityPolicy` 한 곳에 모여 있습니다.
- **문헌 근거가 있는 값**은 출처를 명시했습니다. 예: SNR 30 dB와 19 kHz (Deliyski et al. 2005).
- 연구가 **UNVERIFIED HYPOTHESIS**로 표시한 값은 **잠정값(provisional)**으로 표기했습니다. 예: CPPS f0 상한, 역필터링 f0 상한, 포먼트 분해 비율, 컨텍스트 제외 창.
- 잠정값은 `gyeol.verification.thresholds.operating_threshold`로 측정한 운영 임계값으로 교체해야 합니다.

```python
from gyeol import Engine, EngineConfig, ValidityPolicy
engine = Engine(EngineConfig(policy=ValidityPolicy(cpps_max_f0=650, min_snr_db=35)))
```

nuisance 추정(`rep.nuisance`)은 T_voice에 절대 들어가지 않고 마스크만 결정합니다.
- **SNR**: 유성 구간 vs 휴지 구간. 프레임 단위 SNR도 함께 사용합니다.
- **유효 대역폭**: 유성 LTAS vs 잡음 LTAS
- **코덱 의심**: 저역통과 절벽이나 손실 압축 컨테이너
- **클리핑**
- **AGC 펌핑**
- **노이즈 게이트**
- **T60**: 휴리스틱

기본값으로 **디노이징과 음성 향상은 하지 않습니다.** 향상 처리가 음향 특징을 잡음 입력보다 더 왜곡시킬 수 있기 때문입니다(연구 §2.3).

## 한국어 컨텍스트

```python
from gyeol.context import lyrics_to_syllables
[s.laryngeal_class for s in lyrics_to_syllables("국밥 좋다")]
# ['lenis', 'fortis', 'fortis', 'aspirated']   ← 표면 발음 [국빱 쪼타] 기준
```

- 규칙 기반 표면 발음 변환: 연음, 경음화, 격음화, ㅎ 탈락, 비음화, 겹받침. 선택적으로 g2pK도 쓸 수 있습니다(`use_g2pk=True`).
- 정렬 방식은 두 가지이며, 어느 쪽을 썼는지 `alignment_source`에 기록됩니다.
  - MFA TextGrid (`words`/`phones` 티어)
  - 휴리스틱 "음표당 1음절"
- 격음·경음 직후 창은 피치 정확도와 발성 집계에서 제외합니다. 경음 뒤의 압착 발성을 긴장(strain)으로 오인하지 않기 위해서입니다.
- `ContextNormalizer`: 가수별로 컨텍스트 조건부 기댓값을 뺀 잔차를 만듭니다. 원시값은 그대로 보존되므로 트로트의 의도적 압착 같은 스타일 정보는 사라지지 않습니다.

## 선택 구성요소

- **분리**: `CallableSeparator`(BS-/Mel-RoFormer 추론 함수 래핑), `DemucsSeparator`, `BackingTrackCanceller`(반주 음원을 아는 노래방 환경)
  - `Engine(separator=a, cross_separator=b)`처럼 두 분리기를 주면 둘의 SI-SDR 일치도가 분리 품질 추정치가 됩니다.
- **기기 EQ**: `DeviceProfile.fit(reference, device, sr)`는 동시 녹음에서 최소위상 역필터와 기기 고역통과 코너를 구합니다. `LTASNormalizer`는 블라인드 방식의 대안입니다.
- **잔차 z**: `gyeol.residual.torch_model`
  - 구성: VIB 병목, 기울기 반전(GRL) nuisance 헤드, 가수 대조 손실, 코어 조건부 멜 재구성
  - 학습된 가중치는 **포함하지 않습니다.** 학습에는 §5.ii의 다기기 동시 녹음 코퍼스가 필요합니다.
  - 학습 후 `Engine(residual_encoder=TorchResidualEncoder(model))`로 연결합니다.
- **추적기 교체**: `PitchTracker` 프로토콜을 따르는 RMVPE, SwiftF0 등을 `Engine(trackers=[...])`로 넣을 수 있습니다.

## 검증 도구 (`gyeol.verification`, 연구 §5)

- `stats`
  - ICC(1,1)/(2,1)/(3,1)과 부트스트랩 신뢰구간, Koo & Li 해석
  - Bland–Altman, SEM, MDC95, 가수 내/가수 간 분산비
  - 우연 수준 이항 검정, EER, TOST, Holm 보정
- `degrade`: 잡음(목표 SNR), 합성 RIR(T60/DRR), 대역 제한, 코덱(ffmpeg), AGC, 클리핑, 기기 응답, 반주 혼합, 표준 조건 그리드
- `thresholds.operating_threshold`: 오차–nuisance 곡선의 95% 상한이 MDC를 넘는 지점을 찾습니다. 단조 분위수로 근사하므로 GAM의 보수적 대용입니다.
- `invariance_report`: 가수 × 조건 표에서 차원별로 ICC, 편향, MDC, 수용 판정(ICC ≥ 0.9이고 |편향| < MDC)을 계산합니다.

## 합성 음성 기반 확인 결과 (`pytest`, 73개 테스트)

정답을 알고 있는 합성 가창 모음(`gyeol.synth`)으로 확인했습니다.

| 항목 | 결과 |
|---|---|
| f0 (90–1000 Hz, 비브라토 ±50 cents) | 중앙 오차 1.4–3.5 cents |
| 비브라토 속도 / 폭 | ±0.2 Hz / ±20% 이내 |
| 포먼트 F1–F3 (f0 110–220 Hz) | 대부분 오차 < 4% |
| 포먼트 F2–F3 (f0 660 Hz, 스윕 추정기) | 오차 < 8% |
| H1*–H2* | 개방 지수(OQ) 순서를 보존합니다. f0 120 Hz에서 /a/와 /i/ 차이가 < 2 dB로, 보정 전의 5 dB 이상에서 줄었습니다. |
| CPPS | SNR과 기식성(breathiness)에 대해 단조 감소 |
| NAQ | OQ에 대해 단조 증가 |
| 고 f0 게이팅 | /i/ 500 Hz에서 f0 > F1이면 F1과 H1*–H2*를 자동으로 무효 처리 |
| 속도 | 53초 음원 기준 약 20초 (CPU, 실시간의 0.37배) |

## 연구 대비 구현 범위와 한계

| 연구 권고 | gyeol의 현재 구현 |
|---|---|
| RMVPE / pYIN / SwiftF0 합의 | pYIN / YIN / SHS 합의. 신경망 추적기는 프로토콜로 연결합니다. |
| QCP 역필터링 (IAIF 폴백) | IAIF만 구현했습니다. QCP는 `InverseFilter` 훅으로 연결합니다. |
| QCP-FB + WLP-AME 포먼트 | STE-가중 LP와 Burg를 구현했습니다. QCP-FB와 AME는 미구현입니다. |
| WORLD D4C 비주기성 | D4C에서 착안한 결정론적 하모닉 빗 추정기 |
| BS-/Mel-RoFormer + HTDemucs | 어댑터만 제공하며 모델 가중치는 포함하지 않습니다. |
| ACE 벤치마크 T60/DRR/C50 추정기 | T60은 휴리스틱이고, DRR/C50은 `None`입니다(훅 제공). |
| 코덱 분류기 | 대역 절벽과 컨테이너 기반 휴리스틱 |
| MFA 한국어 (가창 적응) | TextGrid를 읽기만 합니다. 정렬 자체는 외부 MFA가 수행합니다. |
| 잔차 z | 아키텍처와 손실만 있고 학습된 가중치는 없습니다. |
| GAM 기반 임계값 | 단조 분위수(PAVA) 근사 |

- CPPS 절대값은 Praat이나 ADSV와 구현이 달라서 **임상 cutoff와 직접 비교할 수 없습니다.**
- 모든 수치 검증은 합성 음성에서 수행했습니다. 실제 녹음의 유효 영역은 연구 §5.ii의 다기기 동시 녹음 코퍼스로 측정해야 합니다. 특히 분리기가 CPPS·HNR·H1–H2에 주는 편향을 확인하는 Test S가 필요합니다.

## 라이선스

MIT
