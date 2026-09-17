// 7-day periodic plan for the bachelor's thesis, autumn 2026.
// Reproduces the department form "2026 намрын үечилсэн төлөвлөгөө" so the plan can be
// regenerated from data instead of edited in Word.

#set page(paper: "a4", flipped: true, margin: (x: 1.5cm, y: 0.9cm))
#set text(font: "Times New Roman", size: 10pt, lang: "mn")
#set par(leading: 0.5em)

#let mn-title = "Трансформерт графын алгоритм сургахад юу шаардагдах вэ"
#let en-title = "What it costs to make a graph algorithm learnable by a transformer"

// (group, ((sub-task, (weeks…)), …))
#let tasks = (
  ("Судалгаа", (
    ("Ижил төстэй болон суурь ажлуудыг судлах", (1, 2, 3)),
    ("Судалгааны асуултыг тодорхойлж, онолын суурийг нэгтгэн бичих", (2, 3, 4)),
  )),
  ("Өгөгдлийн шинжилгээ", (
    ("Өгөгдлийн давуу болон сул талуудад анализ хийн статистик дөт замуудыг илрүүлэх", (2, 3)),
    ("Дөт замуудыг хаасан өгөгдөл үүсгэж, алдагдлын төрөл бүрийг хэмжсэн суурь үзүүлэлтээр баталгаажуулах", (3, 4, 5)),
  )),
  ("Туршилтын орчин", (
    ("Туршилтын орчныг бүрдүүлж, мэдэгдэж буй лавлах үр дүнтэй харьцуулан баталгаажуулах", (4, 5)),
    ("Туршилтын төлөвлөгөөг урьдчилан бүртгэх (тохиргоо, давталт, зогсоох дүрэм)", (5,)),
  )),
  ("Үндсэн туршилтууд", (
    ("Загварын нөөц (гүн, өргөн, өгөгдлийн хэмжээ) хувьсгасан хяналттай туршилт, онолын таамаглалтай харьцуулах", (5, 6, 7, 8)),
    ("Хяналтын хэлбэр хувьсгасан туршилт; үр дүнгийн ерөнхийлөлтийг хоёр дахь алгоритм дээр шалгах", (7, 8, 9, 10)),
  )),
  ("Үр дүнгийн боловсруулалт, бичих ажил", (
    ("Үр дүнг боловсруулж найдвартай байдлыг үнэлэн (давталт, хяналт), график, хүснэгт бэлтгэх", (8, 9, 10, 11)),
    ("Бүлгүүдийг туршилттай зэрэгцүүлэн бичиж, бүрэн ноорогийг удирдагчид өгөх", (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
  )),
  ("Хамгаалалт", (
    ("Илтгэл бэлтгэж урьдчилсан хамгаалалтад оролцох", (12, 13)),
    ("Шүүмж, зөвлөмжийн дагуу засвар хийж жинхэнэ хамгаалалтад оролцох", (14, 15, 16)),
  )),
)

#let milestones = (
  "6": "Явц 1", "9": "Явц 2", "13": "Урьдчилсан хамгаалалт", "15": "Шүүмж", "16": "Жинхэнэ хамгаалалт",
)
#let n-weeks = 16
#let plan-fill = rgb("#C6D9F1")   // the pale blue the department template uses

// ---- header ----
#align(right)[
  Батлав. \
  МКУТ-ийн эрхлэгч: #box(width: 4cm, repeat[.]) /Профессор. Ч.Алтангэрэл/ \
  2026 оны 09 сарын #box(width: 0.8cm, repeat[.]) нд
]

#v(0.3em)
#align(center)[
  *Монгол нэр: #mn-title* \
  *(англи нэр) #en-title* \
  Сэдэвт бакалаврын судалгааны ажлын \
  *7 хоногийн үечилсэн төлөвлөгөө*
]

#v(0.4em)
Хугацаа: 2026.09.07-оос 2026.12.28 хүртэл

#v(0.4em)

// ---- table ----
#let week-cols = range(1, n-weeks + 1).map(w => if str(w) in milestones { 1.6cm } else { 0.62cm })

#let header-cell(w) = {
  set text(size: 8pt)
  align(center + horizon)[
    #str(w)
    #if str(w) in milestones [ \ #milestones.at(str(w)) ]
  ]
}

#let rows = ()
#for (i, (group, subs)) in tasks.enumerate() {
  for (j, (sub, weeks)) in subs.enumerate() {
    let row = ()
    if j == 0 {
      row.push(table.cell(rowspan: subs.len(), align: center + horizon)[#str(i + 1)])
      row.push(table.cell(rowspan: subs.len(), align: left + horizon)[#group])
    }
    row.push(table.cell(align: left + horizon, text(size: 8.5pt, sub)))
    for w in range(1, n-weeks + 1) {
      row.push(table.cell(fill: if w in weeks { plan-fill } else { none })[])
    }
    rows += row
  }
}

#table(
  columns: (0.7cm, 2.6cm, 6.4cm, ..week-cols),
  stroke: 0.5pt,
  inset: 2.5pt,
  table.header(
    table.cell(align: center + horizon)[№],
    table.cell(colspan: 2, align: center + horizon)[
      #align(right)[Долоо хоног] #v(-0.3em) #align(left)[Хийх ажил]
    ],
    ..range(1, n-weeks + 1).map(header-cell),
  ),
  ..rows,
)

#v(0.4em)
#text(size: 8.5pt)[
  Тайлбар: Төслийг хэрэгжүүлэх төлөвлөгөөг 7 хоногийн давтамжтайгаар хийж тод хараар
  будаж тэмдэглэнэ. Хийх ажил дэд хэсэгтэй байвал уг ажилд зарцуулах хугацааг хуваан
  төлөвлөж болно. Ажлын эхлэх төгсөх хугацаа хоорондоо давхцаж болно. Ажлын гүйцэтгэлийг
  дүгнэж тэмдэглэгээ хийх боломжтой байхаар "7 хоног" баганыг үүсгэнэ.
]

#v(0.4em)
#align(right)[
  Зөвшөөрсөн: Удирдагч багш #box(width: 4cm, repeat[.]) /...багшийн нэр.../ \
  Боловсруулсан: Оюутан #box(width: 4cm, repeat[.]) /...анги, нэр.../ \
  Оюутны ID #box(width: 3cm, repeat[.]) \
  Холбогдох утас: #box(width: 3cm, repeat[.])
]
