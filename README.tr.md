# Fuzuli

Fuzuli, divan şiirlerinin aruz ölçüsünü tanıyan bir derin öğrenme modelidir. Aruz ölçüleri basit bir dil yapısındadır ve problem aslında bir çeviri problemlemidir. 
Bu sebeple model, nöral makine çevirisi için kullanılan [encoder-decoder](https://github.com/tensorflow/nmt) yapısı baz alınarak, *_seq2seq_* bir model gerçeklendi. 
Encoder-Decoder modelin araştırma makalesine [şuradan](https://arxiv.org/pdf/1609.08144.pdf) ulaşabilirsiniz.

### Data
Veriseti, aruz ölçüleri bilinen divan şiirlerinden oluşmaktadır. Bu verisetindeki şiirler kültür bakanlığı tarafından [yayınlanan](http://ekitap.kulturturizm.gov.tr/TR,78354/divanlar.html) açık
divan antolojilerinden toplandı ve düzenlendi.
```
~700 şiir
~7000 mısra
58 aruz ölçüsü
```

Kullanılan antolojiler
```
İZAHLI DİVAN ŞİİRİ ANTOLOJİSİ, Necmettin Halil Onan
ÂHÎ Divanı, Mustafa S. KAÇALİN
DÎVÂN-I YÛNUS EMRE, Dr. Mustafa Tatcı
BÂKÎ DÎVÂNI, Prof. Dr. Sabahattin KÜÇÜK
```

### Training
Kullanım
```
python nmt.py ./../data/
```

**Note**: Her test adımında checkpoint oluşturulduğundan, istenilen zamanda durdurulabilir.

Örnek:
```
Loss :  151.909
---Test---
 -Predictions- 
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffFt</go>tnnlututututututuMMMMMMMM'a11111nnenlelelue
MMffâtnnlliiüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
MMffâtnnl<go>liüliüliülinlinliuuûeîuîîutututututututM
 -Targets- 
Mef'ûlü Fâilâtü Mefâîlü Fâilün</go>                   
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Müstef'ilün Müstef'ilün</go>                          
Mefâîlün Mefâîlün Mefâîlün Mefâîlün</go>              
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Fâ'ilâtün Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>            
Mefâ'îlün Mefâ'îlün Fe'ûlün</go>                      
Müfte'îlün Mefâ'îlün Müfte'îlün Mefâ'îlün</go>        
Mef'ûlü Fâ'ilâtü Mefâ'îlü Fâ'ilün</go>                
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Feilâtün Fâilâtün Feilâtün Feilâtün Feilün Fa'lün</go>
Müstef'ilün Müstef'ilün Müstef'ilün Müstef'ilün</go>  
Mefâ'îlün Mefâ'îlün Mefâ'îlün Mefâ'îlün</go>          
Fâ'ilâtün Fâ'ilâtün Fâ'ilün</go>                      
Mef'ûlü Fâilâtü Mefâîlü Fâilün</go> 
```

### Demo
Demo, kaydedilen checkpointler ile çalışır. Çalıştırmadan önce dosyadan checkpoint ismini değiştirmeyi unutmayın.

Kullanım
```
python nmt_demo.py ./../data/
```

### Versiyonlar
```
tensorflow                1.4.1
numpy                     1.11.3
pandas                    0.22.0
```


