"""Synthetic, fictional content for Phase 3 fixtures.

No real person or company: names are placeholders, e-mail addresses use the reserved example.com
domain, phone numbers and identifiers are made up. Kept as plain data so it can be reviewed
without opening binary files.
"""

CONTRACT_EN = {
    "title": "Employment Contract",
    "intro": "This employment contract is entered into between the parties listed below.",
    "fields": [
        ("Employer Name", "Example Tech Solutions LLC"),
        ("Employee Name", "Jordan Sample"),
        ("Nationality", "Saudi"),
        ("Employee ID", "EMP-00123"),
        ("Job Title", "Data Analyst"),
        ("Work Location", "Riyadh"),
        ("Contract Type", "Fixed-term"),
        ("Start Date", "15/10/2026"),
        ("End Date", "14/10/2027"),
    ],
    "compensation_heading": "Compensation",
    "compensation_rows": [
        ("Basic Salary", "12,000 SAR per month"),
        ("Housing Allowance", "3,000 SAR per month"),
        ("Transportation Allowance", "800 SAR per month"),
        ("Mobile Allowance", "200 SAR per month"),
    ],
    "clauses": [
        ("Probation Period", "The Employee shall be subject to a probation period of ninety (90) days from the start date."),
        ("Working Hours", "Working Hours: 8 hours per day, 40 hours per week."),
        ("Working Days", "The working days are Sunday to Thursday."),
        ("Annual Leave", "The Employee is entitled to paid annual leave of thirty (30) days per contract year."),
        ("Notice Period", "Either party may terminate this contract by giving sixty (60) days written notice."),
        ("Termination", "This contract may be terminated in accordance with the Saudi Labor Law. "
                        "Termination during the probation period does not entitle either party to compensation."),
    ],
}

CONTRACT_AR = {
    "title": "عقد عمل",
    "intro": "تم الاتفاق بين الطرفين المذكورين أدناه على ما يلي:",
    "fields": [
        ("الطرف الأول (صاحب العمل)", "شركة سند التجريبية للتقنية"),
        ("اسم العامل", "ريم الاختبار"),
        ("الجنسية", "سعودية"),
        ("الرقم الوظيفي", "EMP-00456"),
        ("المسمى الوظيفي", "محاسبة"),
        ("مكان العمل", "جدة"),
        ("نوع العقد", "محدد المدة"),
        ("تاريخ بداية العقد", "01/11/2026"),
        ("مدة العقد", "سنة واحدة"),
    ],
    "compensation_heading": "الأجر والبدلات",
    "compensation_rows": [
        ("الراتب الأساسي", "9,500 ريال سعودي شهرياً"),
        ("بدل السكن", "٢٥٠٠ ريال"),
        ("بدل النقل", "25% من الراتب الأساسي"),
        ("بدل طعام", "300 ريال"),
    ],
    "clauses": [
        ("البند الرابع: فترة التجربة", "يخضع العامل لفترة تجربة مدتها تسعون يوماً تبدأ من تاريخ مباشرة العمل."),
        ("البند الخامس: ساعات وأيام العمل", "ساعات العمل: ثماني ساعات يومياً.\nأيام العمل: من الأحد إلى الخميس."),
        ("البند السادس: الإجازات", "يستحق العامل إجازة سنوية مدتها واحد وعشرون يوماً مدفوعة الأجر."),
        ("البند السابع: إنهاء العقد", "يجوز لأي من الطرفين إنهاء هذا العقد بإشعار كتابي مدته ستون يوماً."),
    ],
}

CV_EN = {
    "name": "Jordan Sample",
    "contact": "Email: jordan.sample@example.com | Phone: +966 55 000 1234",
    "location": "Location: Riyadh, Saudi Arabia",
    "summary_heading": "Professional Summary",
    "summary": "Data analyst with a focus on reporting and dashboard development.",
    "skills_heading": "Skills",
    "skills": ["Python", "SQL", "Power BI", "Data Visualization"],
    "experience_heading": "Work Experience",
    "experience": [
        ("Data Analyst at Example Analytics Co.", "Jan 2021 – Present",
         ["Built monthly sales dashboards.", "Automated data quality checks."]),
        ("Junior Analyst at Sample Retail Group", "2018 – 2020", ["Prepared weekly performance reports."]),
    ],
    "education_heading": "Education",
    "education": ["Bachelor of Science in Computer Science, Example University, 2014 – 2018"],
    "certifications_heading": "Certifications",
    "certifications": ["Certified Data Professional (2022)"],
    "languages_heading": "Languages",
    "languages": ["Arabic (Native)", "English (Fluent)"],
}

CV_AR = {
    "name": "ريم الاختبار",
    "contact": "البريد الإلكتروني: reem.test@example.com",
    "phone": "الجوال: 0550001234",
    "location": "المدينة: جدة",
    "summary_heading": "نبذة مختصرة",
    "summary": "محاسبة لديها خبرة في إعداد التقارير المالية.",
    "skills_heading": "المهارات",
    "skills_line": "إعداد القوائم المالية، تحليل التكاليف، Excel",
    "experience_heading": "الخبرات العملية",
    "experience": [
        ("محاسبة في شركة مثال للتجارة", "2019 - حتى الآن", ["إعداد التقارير الشهرية."]),
    ],
    "education_heading": "التعليم",
    "education": ["بكالوريوس محاسبة، جامعة المثال، 2015 - 2019"],
    "certifications_heading": "الشهادات",
    "certifications": ["شهادة المحاسب المعتمد التجريبية (2021)"],
    "languages_heading": "اللغات",
    "languages": ["العربية (اللغة الأم)", "الإنجليزية (متقدم)"],
}
