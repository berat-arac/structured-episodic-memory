# Project B Pong — Trainable Oracle-Free SEM

Bu paket, eski `project_b_pong_v1_adaptive` tabanından yeniden kurulmuş oracle-free SEM Pong sürümüdür. Amaç iki ayrı şeyi gerçekten öğrenilmiş hale getirmektir:

1. **Motor skill:** SEM topa yetişip geri döndürmeyi deneyimden öğrenir.
2. **Opponent tactics:** SEM rakibin hangi contact bölgelerinde zorlandığını gözlemler, bir sonraki incoming top için bir contact intent seçer ve goal-conditioned motor belleğiyle o bölgeyi gerçekleştirmeye çalışır.

Paketin içinde ayrıca 40-run eğitimden üretilmiş örnek checkpoint vardır. İstersen direkt oynayabilir, istersen kendi makinen ve seed'inle sıfırdan yeniden eğitebilirsin.

## Hızlı başlangıç

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kendi makinen üzerinde 40 run eğitim:

```powershell
python train_sem.py
```

Ardından eğittiğin SEM ile oyna:

```powershell
python play_human_vs_trained_sem.py
```

Paketle gelen hazır checkpoint ile eğitim beklemeden de ikinci komutu doğrudan çalıştırabilirsin.

## 40 run tam olarak ne yapıyor?

Varsayılan eğitim:

```text
40 run
55 SEM incoming outcome / run
2200 toplam motor outcome
her run farklı environment seed
her run farklı coach seed
6 farklı güçlü coach stili döngüsel kullanılır
```

Coach stilleri:

- `center`
- `spinner`
- `top_bias`
- `bottom_bias`
- `mixed`
- `alternator`

Coach'lar yalnız eğitim verisi üretir. **SEM'in bir parçası değildir ve bilgileri SEM'e verilmez.**

Curriculum:

```text
ilk %35   neutral intent      -> önce topa yetişmeyi öğren
sonraki %55 random intent     -> farklı contact hedeflerini gerçekleştirmeyi öğren
son %10   strategy intent     -> opponent memory -> motor intent zincirini çalıştır
```

Her 5 run'da checkpoint yazılır. Eğitim sonunda:

```text
checkpoints/sem_40runs.json.gz
results/training_40runs.json
```

oluşur.

## Farklı seed ile yeni SEM

Örneğin:

```powershell
python train_sem.py `
  --seed 12345 `
  --checkpoint checkpoints/sem_seed12345.json.gz `
  --report results/train_seed12345.json
```

Sonra:

```powershell
python play_human_vs_trained_sem.py --checkpoint checkpoints/sem_seed12345.json.gz
```

Başka bir seed için aynı komutu farklı değerle çalıştırabilirsin. Böylece aynı mimarinin farklı eğitim geçmişlerinden çıkan SEM'lerini karşılaştırabilirsin.

## Neler düzeltildi?

### 1. Frame başına rastgele karar kaldırıldı

Eski rebuild her 60 FPS frame'de yeni action seçebildiği için paddle çok titreşimliydi. Yeni sürümde motor kararı birkaç frame tutulur. Exploration yalnız **yeni motor kararı** sırasında örneklenir.

### 2. Generic actuator smoothing eklendi

Öğrenilmiş komut doğrudan `-1050 -> +1050` şeklinde zıplamaz. Paddle velocity yumuşak biçimde yeni komuta yaklaşır.

Bu smoothing topun nerede olduğunu söylemez. Yalnız motor çıkışına inertia benzeri süreklilik verir.

### 3. Motor switching cost eklendi

İki action'ın öğrenilmiş değeri birbirine çok yakınsa SEM gereksiz yere ters yöne dönmek yerine mevcut motor intent'ini korur. Yeni action gerçekten daha iyi öğrenilmiş değere sahipse yön değiştirebilir.

### 4. Eligibility trace frame yerine motor-decision seviyesine taşındı

Bir hit sonucunun yüzlerce çelişkili frame action'ını ödüllendirmesi engellendi. Survival trace ve aiming trace ayrı tutulur.

### 5. Survival ve aiming motor belleği ayrıldı

`motor`:

- temel amaç topa yetişmek,
- hit `+1`, miss `-1.15`,
- current-frame state -> velocity action ilişkisini öğrenir.

`aim_motor`:

- strategy katmanından gelen contact intent'i görür,
- hit anında gerçek contact ile intent arasındaki hatadan sparse reward alır,
- temel motor skill'i bozmadan residual action preference öğrenir.

### 6. Opponent memory artık motora bağlı

SEM rakibe attığı toplarda gerçek contact bölgesini ve rakibin cevabını kaydeder.

Rakip topu geri döndürdüğünde off-centre contact bir miktar difficulty sinyali sağlar. Rakip topu kaçırırsa en güçlü outcome gelir.

Sonraki incoming başında strategy memory bir contact intent seçer. Bu intent `aim_motor` için goal olur.

Yani artık zincir gerçekten şudur:

```text
rakibi gözle
    -> hangi contact zone daha etkili öğren
    -> yeni contact intent seç
    -> motor memory o intent'i gerçekleştirmeye çalışsın
    -> gerçek sonucu tekrar opponent memory'ye yaz
```

## Hard-coded olanlar / olmayanlar

### Hard-coded veya engineered

Bunları özellikle saklamıyoruz:

- Pong fizik motoru
- gözlemde `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, paddle konumları
- topun SEM'e doğru gelip gelmediğini ayıran `ball_vx > 0` gate'i
- state bin'leri ve feature tasarımı
- discrete motor action vocabulary
- hit/miss sparse reward tanımı
- actuator smoothing ve motor switching cost
- contact intent için current-frame geometric goal-error feature'ı
- eğitim coach'ları ve curriculum

Goal-error feature yalnız **mevcut frame'deki** ball-paddle geometry ile istenen contact arasındaki farkı temsil eder. Hangi action'ın yapılacağını söylemez; action ilişkisi deneyimden öğrenilir.

### SEM içinde olmayanlar

- gelecekteki paddle-line intersection hesabı
- duvar sekmelerini ileri sarıp future-y bulma
- time-to-impact hesabı
- `top yukarıdaysa paddle yukarı` gibi elle yazılmış hareket kuralı
- neural network
- backprop / autodiff
- PyTorch / TensorFlow / JAX

## Oyun sırasında gerçekten öğrenmeye devam ediyor mu?

Evet.

Checkpoint yalnız başlangıç motor becerisidir. Human game sırasında:

- motor hit/miss belleği güncellenmeye devam eder,
- **rakip strategy memory varsayılan olarak sıfırdan başlar**, yani coach'lara karşı öğrendiği rakip modelini sana taşımayız,
- senin return/miss davranışına göre contact zone değerleri canlı güncellenir,
- current intent debug panelinde görülebilir.

Çıkınca iki dosya yazılır:

```text
results/last_human_match.json
checkpoints/sem_after_human.json.gz
```

İkincisi sana karşı adapte olmuş SEM checkpoint'idir.

## Human game kontrolleri

```text
W / S veya Up / Down   insan paddle
R                      yeni maç, TÜM öğrenme korunur
O                      yalnız opponent/tactic memory sıfırlanır, motor korunur
X                      pretrained checkpoint'e geri dön, opponent memory sıfırla
N                      yeni round, öğrenme korunur
D                      debug panel aç/kapat
ESC                    kaydet ve çık
```

Debug panelde özellikle şunlara bak:

- `human-session motor`: sana karşı kaç motor outcome gördü
- `current intent`: bu incoming için seçtiği contact hedefi
- `last actual contact`: gerçekte nereden vurdu
- `best observed zone`: sana karşı şu ana kadar en değerli gördüğü bölge
- contact tablosundaki `fast / slow / n`: taktik belleğin nasıl değiştiği

Birkaç shot'ta tablo çok gürültülü olur. Daha uzun oynadıkça `fast` değerlerin ve tercihlerin rakibin davranışına göre değişmesi beklenir.

## Validation

```powershell
python run_validation.py
```

Çıktı:

```text
results/validation_report.json
```

Final paket oluşturulurken elde edilen doğrulama:

```text
status: PASS
fresh-seed miss reduction: ~72.2%
pretrained mean hit rate: ~97.9%
pretrained motor sign flips: ~25 / 1000 incoming frame
random contact-intent evaluation hit rate: ~96.7%
intent -> actual contact correlation: ~0.40
checkpoint roundtrip: PASS
future-path source audit: PASS
```

Ayrı 40-run training report'unda:

```text
2200 motor outcomes
2108 hits / 92 misses
training lifetime hit rate: ~95.8%
held-out evaluation hit rate: ~97.4%
intent -> actual contact correlation: ~0.44
```

Bu sayılar verilen seed ve coach curriculum'u için ölçülmüştür; başka seedlerde birebir aynı sonuç garanti edilmez.

## Bilimsel claim sınırı

Savunulabilir ifade:

> A non-neural memory-based Pong agent learns paddle interception from sparse hit/miss outcomes, learns goal-conditioned contact control from sparse hit-contact error, and adapts fast/slow opponent-response memory online, without an analytical future-trajectory controller.

İddia etmemek gerekenler:

- blank-slate pixel-level Pong learning
- yeni genel RL paradigması
- biologically realistic brain simulation
- SOTA Pong agent
- trajectory-free olmanın feature engineering olmadığı anlamına geldiği
- contact intent'in kusursuz uygulandığı

Contact control ölçülebilir biçimde intent'e bağlıdır ama kusursuz değildir; validation'daki correlation bu yüzden özellikle raporlanır.

## Motor miss audit (diagnostic-only)

Human-vs-SEM oyununda her motor kararı artık yalnızca teşhis amacıyla geçici olarak kaydedilir. Bu audit katmanı action seçimine, reward'a veya learning update'ine müdahale etmez.

Bir motor miss olduğunda son 12 karar için şunlar saklanır:

- raw current-frame ball/paddle observation,
- discretized SEM state features,
- seçilen motor action,
- kararın explore mu exploit mi olduğu ve epsilon,
- base/aim value ve action-margin,
- mevcut state-action için görülen evidence miktarı,
- current contact intent.

Oyun sırasında debug panelinde `last miss cause` görünür. Bu etiket bir **diagnostic heuristic**tir; nedenselliğin matematiksel kanıtı değildir. Olası etiketler: `recent_exploration`, `low_evidence_state`, `ambiguous_policy_values`, `high_speed_generalization`, `learned_policy_error_or_timing`.

Oyundan `ESC` ile çıkınca iki dosya oluşur:

```text
results/last_human_match.json
results/last_human_miss_audit.json
```

Son miss'leri terminalde okunabilir biçimde görmek için:

```bash
python inspect_last_miss.py
```

Bu özellik özellikle "SEM neden o anda ters yöne gitti?" sorusunu source/state/action logu üzerinden incelemek içindir; yeni bir Pong kuralı veya trajectory controller eklemez.

## Gizli rakip-davranış switch testi (A → B → A)

Bu paket ayrıca SEM'in rakip/çevre davranışı değiştiğinde strategy memory'sini yeniden ağırlıklandırıp ağırlıklandırmadığını test eden kontrollü bir benchmark içerir:

```bash
python run_opponent_switch_test.py
```

Protokol:

```text
A1 = weak_up
B  = weak_down
A2 = weak_up
```

Her faz varsayılan olarak 100 strategy outcome sürer ve test 6 bağımsız seed üzerinde tekrarlanır.

Önemli ayrım: **SEM'e faz adı, switch zamanı veya coach modu verilmez.** Motor/aim belleği pretrained checkpoint'ten yüklenir; opponent/strategy memory yalnız test başında sıfırlanır ve A1 → B → A2 arasında hiçbir memory reset yapılmaz. Switch yalnız benchmark rakibinin response contingency'sini değiştirir.

Benchmark rakibi kontrollü olarak, SEM'den ayrılan topun yukarı/aşağı trajectory family'sinden birine karşı daha gecikmeli ve düşük güçlü cevap verir. Bu bir insan modeli değildir; amaç, reward landscape gizlice tersine döndüğünde fast/slow strategy memory'nin online adaptasyonunu ölçmektir.

Paket hazırlanırken elde edilen 6-seed sonuç:

```text
                    first 20 target-intent   last 20 target-intent
A1 weak_up                 65.8%                    94.2%
B  weak_down               45.8%                    87.5%
A2 weak_up                 53.3%                    93.3%
```

Yani B switch'inden hemen sonra SEM bir süre A'dan kalan tercihi sürdürür; yeni sonuçları gördükçe tercih dağılımı B'nin vulnerability'sine kayar. A'ya geri dönüldüğünde aynı desen ters yönde tekrar görülür.

Kontrollü contingency'nin gerçekten farklı reward ürettiği de ölçülür. Fazlar boyunca hedef actual-contact family'sinde opponent-miss oranı yaklaşık %42–48 iken diğer contact family'lerinde yaklaşık %4–7 bandındadır. Bu nedenle adaptation yalnız bir etiket/ranking oyunu değildir; seçimlerin gözlenen sonucu farklıdır.

Tam rapor:

```text
results/opponent_switch_report.json
```

Daha kısa tek-seed test:

```bash
python run_opponent_switch_test.py --seeds 1 --episodes-per-phase 80
```

Bu testten savunulabilecek sonuç: **SEM, switch bilgisini almadan, sonuç istatistiği değişen bir rakibe karşı contact-intent tercihlerini online olarak yeniden ağırlıklandırabiliyor.** Bu kontrollü benchmark, kişi tanıma veya genel insan-rakip modelleme iddiası değildir.
