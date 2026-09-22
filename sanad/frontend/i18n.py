"""Interface language for Sanad: English by default, Arabic fully supported.

Presentation only. Every visible label in the interface comes from the table below, so the same
screens, the same brand and the same workflows serve both languages; only the direction and the
wording change. Text that the backend wrote (findings, explanations, recommendations, article text)
is never translated here - it is shown exactly as it was returned.
"""

from __future__ import annotations

LANGUAGES = {"en": "EN", "ar": "العربية"}
DEFAULT_LANGUAGE = "en"


def is_rtl(lang: str) -> bool:
    return lang == "ar"


def t(key: str, lang: str = DEFAULT_LANGUAGE, **values) -> str:
    """One label, in the chosen language. Missing keys return the key, never an exception."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANGUAGE, key)
    return text.format(**values) if values else text


def _s(en: str, ar: str) -> dict[str, str]:
    return {"en": en, "ar": ar}


STRINGS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ shell
    "app.name": _s("Sanad", "سند"),
    "app.tagline": _s("Your guide to fair work", "دليلك إلى عملٍ عادل"),
    "nav.section": _s("Menu", "القائمة"),
    "nav.home": _s("Home", "الرئيسية"),
    "nav.ask": _s("Ask Sanad", "اسأل سند"),
    "nav.analyze": _s("Analyze Contract", "تحليل العقد"),
    "nav.compare": _s("Compare Contracts", "مقارنة العقود"),
    "nav.salary": _s("Salary Benchmark", "مقارنة الرواتب"),
    "nav.help": _s("Help", "المساعدة"),
    "nav.settings": _s("Settings", "الإعدادات"),
    "nav.language": _s("Language", "اللغة"),
    "action.back": _s("Back", "رجوع"),
    "action.start_over": _s("Start over", "البدء من جديد"),
    "action.continue": _s("Continue", "متابعة"),
    "action.open": _s("Open", "افتح"),
    "action.remove": _s("Remove", "إزالة"),
    "action.change_file": _s("Choose a different file", "اختيار ملف آخر"),
    "common.optional": _s("optional", "اختياري"),
    "common.supported_files": _s("PDF or DOCX", "PDF أو DOCX"),
    "common.file_limit": _s("Up to {size} MB per file", "حتى {size} ميغابايت لكل ملف"),
    "common.selected_file": _s("Selected file", "الملف المختار"),
    "common.evidence": _s("Evidence", "الدليل"),
    "common.sources": _s("Sources", "المصادر"),
    "common.article": _s("Article", "المادة"),
    "common.arabic_text": _s("Arabic (the legal reference)", "النص العربي (المرجع النظامي)"),
    "common.english_text": _s("English (machine translation)", "الإنجليزية (ترجمة آلية)"),
    "common.details": _s("Details", "التفاصيل"),
    "common.notes": _s("Notes and limitations", "ملاحظات وحدود"),
    "common.documents": _s("Uploaded documents", "الملفات المرفوعة"),
    "common.disclaimer": _s("Sanad provides information based on the Saudi Labor Law. It is not legal advice.",
                            "يقدّم سند معلومات مبنية على نظام العمل السعودي، وهي ليست استشارة قانونية."),

    # ------------------------------------------------------------------ home
    "home.hero_title": _s("Sanad", "سند"),
    "home.hero_sub": _s(
        "Understand your employment contract, ask about Saudi labor regulations, compare offers, "
        "and explore salary ranges.",
        "افهم عقد عملك، واسأل عن أنظمة العمل السعودية، وقارن العروض، واستكشف نطاقات الرواتب."),
    "home.services": _s("What can Sanad help you with?", "بماذا يمكن لسند مساعدتك؟"),
    "home.ask_title": _s("Ask Sanad", "اسأل سند"),
    "home.ask_desc": _s("Ask questions about Saudi labor regulations and employment contracts.",
                        "اطرح أسئلتك عن أنظمة العمل السعودية وعقود العمل."),
    "home.ask_cta": _s("Ask a question", "اطرح سؤالاً"),
    "home.analyze_title": _s("Analyze Contract", "تحليل العقد"),
    "home.analyze_desc": _s("Upload an employment contract and get a detailed compliance analysis.",
                            "ارفع عقد العمل واحصل على تحليل تفصيلي لمدى التزامه بالنظام."),
    "home.analyze_cta": _s("Analyze a contract", "حلّل عقداً"),
    "home.compare_title": _s("Compare Contracts", "مقارنة العقود"),
    "home.compare_desc": _s("Compare 2–5 employment contracts side by side.",
                            "قارن بين ٢ إلى ٥ عقود عمل جنباً إلى جنب."),
    "home.compare_cta": _s("Compare offers", "قارن العروض"),
    "home.salary_title": _s("Salary Benchmark", "مقارنة الرواتب"),
    "home.salary_desc": _s("Explore salary ranges based on job title, location, and experience.",
                           "استكشف نطاقات الرواتب حسب المسمى الوظيفي والمدينة وسنوات الخبرة."),
    "home.salary_cta": _s("Check a salary", "افحص راتباً"),
    "home.trust": _s("Every answer shows the regulation it came from.",
                     "كل إجابة تعرض المادة النظامية التي استندت إليها."),

    # ------------------------------------------------------------------ ask
    "ask.title": _s("Ask Sanad", "اسأل سند"),
    "ask.sub": _s("Get clear answers about Saudi labor regulations and employment contracts.",
                  "احصل على إجابات واضحة عن أنظمة العمل السعودية وعقود العمل."),
    "ask.input_label": _s("Your question", "سؤالك"),
    "ask.placeholder": _s("Type your question…", "اكتب سؤالك…"),
    "ask.suggestions": _s("Try one of these", "جرّب أحد هذه الأسئلة"),
    "ask.q1": _s("What is the probation period?", "ما هي مدة فترة التجربة؟"),
    "ask.q2": _s("How many annual leave days do I get?", "كم عدد أيام الإجازة السنوية؟"),
    "ask.q3": _s("What are the rules for part-time work?", "ما أحكام العمل بدوام جزئي؟"),
    "ask.q4": _s("What are the end-of-service benefits?", "ما هي مكافأة نهاية الخدمة؟"),
    "ask.cta": _s("Ask", "اسأل"),
    "ask.empty": _s("Write a question to get started.", "اكتب سؤالاً للبدء."),
    "ask.answer_title": _s("Answer", "الإجابة"),
    "ask.no_answer": _s("Sanad retrieved the relevant regulations but could not write an answer.",
                        "استرجع سند المواد النظامية ذات العلاقة ولم يتمكّن من صياغة إجابة."),
    "ask.contract_note": _s(
        "Questions are answered from the Saudi Labor Law itself. To have your own contract examined, "
        "use Analyze Contract.",
        "تُجاب الأسئلة من نظام العمل السعودي نفسه. ولفحص عقدك الخاص، استخدم «تحليل العقد»."),

    # ------------------------------------------------------------------ analyze
    "analyze.title": _s("Analyze Your Contract", "حلّل عقدك"),
    "analyze.sub": _s("Upload your employment contract and get a detailed analysis based on Saudi labor regulations.",
                      "ارفع عقد عملك واحصل على تحليل تفصيلي وفق أنظمة العمل السعودية."),
    "analyze.step1": _s("Upload contract", "رفع العقد"),
    "analyze.step2": _s("Choose analysis", "اختيار نوع التحليل"),
    "analyze.step3": _s("Results", "النتائج"),
    "analyze.upload_label": _s("Upload your employment contract", "ارفع عقد العمل"),
    "analyze.upload_hint": _s("Drag a file here or choose one from your computer.",
                              "اسحب الملف إلى هنا أو اختره من جهازك."),
    "analyze.choose_title": _s("Do you want to analyze the contract only, or compare it with your CV?",
                               "هل تريد تحليل العقد فقط، أم مقارنته بسيرتك الذاتية؟"),
    "analyze.only_title": _s("Contract Only", "العقد فقط"),
    "analyze.only_desc": _s("Analyze the contract based on Saudi labor regulations.",
                            "تحليل العقد وفق أنظمة العمل السعودية."),
    "analyze.cv_title": _s("Contract + CV", "العقد + السيرة الذاتية"),
    "analyze.cv_desc": _s("Analyze the contract and evaluate its compatibility with your experience and skills.",
                          "تحليل العقد وتقييم مدى توافقه مع خبراتك ومهاراتك."),
    "analyze.cv_upload_label": _s("Upload your CV", "ارفع سيرتك الذاتية"),
    "analyze.role_label": _s("Role you are being hired for", "الوظيفة المتقدَّم لها"),
    "analyze.role_help": _s("Leave empty to use the job title written in the contract.",
                            "اتركه فارغاً لاستخدام المسمى الوظيفي المذكور في العقد."),
    "analyze.skills_label": _s("Key skills required, separated by commas", "المهارات المطلوبة، مفصولة بفواصل"),
    "analyze.cta": _s("Analyze contract", "حلّل العقد"),
    "analyze.cta_cv": _s("Analyze contract and CV", "حلّل العقد والسيرة الذاتية"),
    "analyze.cv_only_switch": _s("I only want to check a CV", "أريد فحص سيرة ذاتية فقط"),
    "analyze.cv_only_title": _s("Check a CV", "فحص سيرة ذاتية"),
    "analyze.cv_only_sub": _s("Upload a CV and the role you are aiming for.",
                              "ارفع السيرة الذاتية والوظيفة التي تستهدفها."),
    "analyze.cv_only_cta": _s("Check CV", "افحص السيرة"),
    "analyze.back_to_contract": _s("Back to contract analysis", "العودة إلى تحليل العقد"),
    "analyze.progress": _s("Analyzing your contract…", "جارٍ تحليل عقدك…"),
    "analyze.p1": _s("Reading contract", "قراءة العقد"),
    "analyze.p2": _s("Extracting contract terms", "استخراج بنود العقد"),
    "analyze.p3": _s("Checking applicable regulations", "مطابقة المواد النظامية"),
    "analyze.p4": _s("Preparing results", "تجهيز النتائج"),
    "analyze.results_title": _s("Contract Analysis", "تحليل العقد"),
    "analyze.summary_title": _s("Analysis summary", "ملخّص التحليل"),
    "analyze.findings_title": _s("Detailed findings", "النتائج التفصيلية"),
    "analyze.compliance_title": _s("Contract Compliance", "التزام العقد بالنظام"),
    "analyze.cv_section": _s("CV Compatibility", "توافق السيرة الذاتية"),
    "analyze.cv_observations": _s("Notes on the CV", "ملاحظات على السيرة الذاتية"),
    "analyze.other_findings": _s("Other terms recorded from the contract ({n})",
                                 "بنود أخرى مسجّلة من العقد ({n})"),
    "analyze.contract_says": _s("The contract says", "العقد ينص على"),
    "analyze.page": _s("Page", "صفحة"),
    "analyze.new": _s("Analyze another contract", "حلّل عقداً آخر"),

    # ------------------------------------------------------------------ compare
    "compare.title": _s("Compare Contracts", "مقارنة العقود"),
    "compare.sub": _s("Upload 2–5 employment contracts and compare their key terms.",
                      "ارفع من ٢ إلى ٥ عقود عمل وقارن بين بنودها الأساسية."),
    "compare.step1": _s("Upload contracts", "رفع العقود"),
    "compare.step2": _s("Your priorities", "أولوياتك"),
    "compare.step3": _s("Results", "النتائج"),
    "compare.slot": _s("Contract {n}", "العقد {n}"),
    "compare.add": _s("Add another contract", "أضف عقداً آخر"),
    "compare.remove_last": _s("Remove last slot", "احذف آخر خانة"),
    "compare.min_note": _s("Upload at least 2 contracts to compare.", "ارفع عقدين على الأقل للمقارنة."),
    "compare.max_note": _s("Up to {n} contracts can be compared at once.",
                           "يمكن مقارنة حتى {n} عقود في المرة الواحدة."),
    "compare.priorities_title": _s("What matters most to you?", "ما الأهم بالنسبة لك؟"),
    "compare.priorities_help": _s(
        "Optional. If the contracts do not clearly outrank each other, Sanad uses your order of "
        "priorities to explain which one fits you better.",
        "اختياري. إذا لم يتفوق أي عقد بوضوح، يستخدم سند ترتيب أولوياتك لتوضيح أيّها أنسب لك."),
    "compare.priorities_order": _s("Selected in order, most important first.",
                                   "تُرتَّب حسب اختيارك، الأهم أولاً."),
    "compare.cta": _s("Compare contracts", "قارن العقود"),
    "compare.progress": _s("Comparing your contracts…", "جارٍ مقارنة عقودك…"),
    "compare.results_title": _s("Contract Comparison", "مقارنة العقود"),
    "compare.recommendation": _s("Recommendation", "التوصية"),
    "compare.preferred": _s("Preferred: {name}", "الأفضل: {name}"),
    "compare.no_preference": _s("No clear preference", "لا تفضيل واضح"),
    "compare.why": _s("Why", "الأسباب"),
    "compare.trade_offs": _s("Trade-offs", "المفاضلات"),
    "compare.table": _s("Side-by-side comparison", "مقارنة جنباً إلى جنب"),
    "compare.table_hint": _s("Scroll sideways to see every contract.", "مرّر أفقياً لرؤية جميع العقود."),
    "compare.risks": _s("Risks and review items", "المخاطر وبنود المراجعة"),
    "compare.new": _s("Compare other contracts", "قارن عقوداً أخرى"),

    # ------------------------------------------------------------------ salary
    "salary.title": _s("Salary Benchmark", "مقارنة الرواتب"),
    "salary.sub": _s("Explore salary ranges for your role based on available market data.",
                     "استكشف نطاقات الرواتب لوظيفتك بناءً على بيانات السوق المتاحة."),
    "salary.step1": _s("Job information", "معلومات الوظيفة"),
    "salary.step2": _s("Results", "النتائج"),
    "salary.contract_label": _s("Upload the contract with the salary to check",
                                "ارفع العقد الذي تريد فحص راتبه"),
    "salary.contract_note": _s(
        "Sanad benchmarks the salary written in a contract, so the contract is required. The details "
        "below refine the search.",
        "يقارن سند الراتب المذكور في العقد، لذا فإن رفع العقد مطلوب. التفاصيل أدناه تحسّن نتائج البحث."),
    "salary.job_title": _s("Job title", "المسمى الوظيفي"),
    "salary.job_title_help": _s("Leave empty to use the job title written in the contract.",
                                "اتركه فارغاً لاستخدام المسمى الوظيفي المذكور في العقد."),
    "salary.city": _s("City", "المدينة"),
    "salary.years": _s("Years of experience", "سنوات الخبرة"),
    "salary.cta": _s("Get salary benchmark", "احصل على المقارنة"),
    "salary.progress": _s("Looking up salary data…", "جارٍ البحث في بيانات الرواتب…"),
    "salary.results_title": _s("Salary Benchmark", "مقارنة الرواتب"),
    "salary.range_title": _s("Estimated salary range", "نطاق الراتب المقدَّر"),
    "salary.contract_salary": _s("Salary in the contract", "الراتب في العقد"),
    "salary.position": _s("Where the contract sits", "موقع العقد من النطاق"),
    "salary.position_below": _s("Below the observed range", "أقل من النطاق المرصود"),
    "salary.position_inside": _s("Inside the observed range", "ضمن النطاق المرصود"),
    "salary.position_above": _s("Above the observed range", "أعلى من النطاق المرصود"),
    "salary.basis": _s("Measured as", "نوع الراتب"),
    "salary.data_quality": _s("Data quality", "جودة البيانات"),
    "salary.evidence": _s("Figures read from the sources", "الأرقام المقروءة من المصادر"),
    "salary.retrieved": _s("retrieved", "تاريخ الاسترجاع"),
    "salary.new": _s("Check another salary", "افحص راتباً آخر"),
    "salary.unavailable": _s("Salary benchmarking is not switched on for this installation.",
                             "خدمة مقارنة الرواتب غير مفعّلة في هذا التركيب."),
    "salary.no_range": _s("No defensible market range could be built from the available sources.",
                          "تعذّر بناء نطاق سوقي موثوق من المصادر المتاحة."),

    # ------------------------------------------------------------------ statuses
    "status.compliant": _s("Compliant", "مطابق"),
    "status.attention": _s("Needs attention", "يحتاج مراجعة"),
    "status.non_compliant": _s("Non-compliant", "غير مطابق"),
    "status.unavailable": _s("Not applicable", "غير منطبق"),
    "status.not_found": _s("Not stated in the contract", "غير مذكور في العقد"),
    "status.document_fact": _s("Contract fact", "معلومة من العقد"),
    "status.check_failed": _s("Could not be checked", "تعذّر الفحص"),
    "status.completed": _s("Completed", "اكتمل"),
    "status.partial": _s("Completed with gaps", "اكتمل مع نواقص"),
    "status.insufficient": _s("Not enough evidence", "أدلة غير كافية"),
    "status.rejected": _s("Request rejected", "طلب مرفوض"),
    "status.rag_error": _s("Regulatory search unavailable", "خدمة البحث النظامي غير متاحة"),
    "status.failed": _s("Something went wrong", "حدث خطأ"),

    # ------------------------------------------------------------------ states
    "state.no_file": _s("No file chosen yet", "لم يتم اختيار ملف بعد"),
    "state.no_file_body": _s("Choose a PDF or DOCX file to continue.", "اختر ملف PDF أو DOCX للمتابعة."),
    "state.unsupported": _s("That file type is not supported", "نوع الملف غير مدعوم"),
    "state.unsupported_body": _s("Sanad reads PDF and DOCX files. Scanned images are not supported.",
                                 "يقرأ سند ملفات PDF وDOCX. الصور الممسوحة ضوئياً غير مدعومة."),
    "state.offline": _s("Sanad is not reachable right now", "تعذّر الوصول إلى سند حالياً"),
    "state.offline_body": _s("The Sanad service is not responding. Please try again in a moment.",
                             "خدمة سند لا تستجيب حالياً. يرجى المحاولة بعد قليل."),
    "state.rag_down": _s("Regulatory search is unavailable", "البحث النظامي غير متاح"),
    "state.rag_down_body": _s(
        "Sanad could not reach the Saudi Labor Law search service, so findings could not be checked "
        "against the regulations.",
        "تعذّر على سند الوصول إلى خدمة البحث في نظام العمل، لذا لم تتم مطابقة النتائج بالمواد النظامية."),
    "state.no_results": _s("Nothing to show yet", "لا توجد نتائج بعد"),
    "state.partial": _s("Some parts of this request could not be completed.",
                        "تعذّر إكمال بعض أجزاء هذا الطلب."),
    "state.error_title": _s("This request could not be completed", "تعذّر إكمال هذا الطلب"),

    # ------------------------------------------------------------------ priorities (labels only;
    # the values themselves are whatever the service reports as supported)
    "priority.basic_salary": _s("Basic salary", "الراتب الأساسي"),
    "priority.stated_pay": _s("Total stated pay", "إجمالي الأجر المذكور"),
    "priority.compliance": _s("Compliance with the law", "الالتزام بالنظام"),
    "priority.cv_compatibility": _s("Fit with my CV", "التوافق مع سيرتي الذاتية"),
    "priority.weekly_working_hours": _s("Weekly working hours", "ساعات العمل الأسبوعية"),
    "priority.daily_working_hours": _s("Daily working hours", "ساعات العمل اليومية"),
    "priority.working_days_per_week": _s("Working days per week", "أيام العمل في الأسبوع"),
    "priority.annual_leave": _s("Annual leave", "الإجازة السنوية"),
    "priority.probation_period": _s("Probation period", "فترة التجربة"),

    # ------------------------------------------------------------------ result rendering
    "ui.regulation": _s("Regulation", "المادة النظامية"),
    "ui.show_evidence": _s("Show the regulation and the evidence", "عرض المادة والدليل"),
    "ui.retrieved_articles": _s("All articles retrieved for this contract ({n})",
                                "جميع المواد المسترجعة لهذا العقد ({n})"),
    "ui.no_findings": _s("No findings were produced for this contract.", "لم تُنتج أي نتائج لهذا العقد."),
    "ui.requirements": _s("Requirements checked", "المتطلبات التي تم فحصها"),
    "ui.no_requirements": _s("No job requirements were available to check the CV against.",
                             "لا توجد متطلبات وظيفية لمطابقة السيرة الذاتية بها."),

    # ------------------------------------------------------------------ help and settings
    "help.title": _s("Help", "المساعدة"),
    "help.sub": _s("What Sanad can do, and what it deliberately will not do.",
                   "ما الذي يستطيع سند فعله، وما الذي يتجنّبه عن قصد."),
    "help.can_title": _s("What Sanad can do", "ما يستطيع سند فعله"),
    "help.can_1": _s("Answer questions about the Saudi Labor Law and show the articles behind the answer.",
                     "الإجابة عن أسئلة نظام العمل السعودي مع عرض المواد التي استند إليها."),
    "help.can_2": _s("Check an employment contract against those regulations, clause by clause.",
                     "فحص عقد العمل مقابل تلك الأنظمة بنداً بنداً."),
    "help.can_3": _s("Compare 2–5 contracts and explain which terms are better and why.",
                     "مقارنة من ٢ إلى ٥ عقود وتوضيح أي البنود أفضل ولماذا."),
    "help.can_4": _s("Benchmark the salary in a contract against published market sources.",
                     "مقارنة الراتب المذكور في العقد بمصادر السوق المنشورة."),
    "help.cannot_title": _s("What Sanad will not do", "ما لا يفعله سند"),
    "help.cannot_1": _s("It never invents an article number, a quotation or a salary figure.",
                        "لا يختلق أبداً رقم مادة أو اقتباساً أو رقم راتب."),
    "help.cannot_2": _s("It does not give legal advice or replace a lawyer.",
                        "لا يقدّم استشارة قانونية ولا يغني عن محامٍ."),
    "help.cannot_3": _s("It does not read scanned images; contracts must be text-based PDF or DOCX files.",
                        "لا يقرأ الصور الممسوحة ضوئياً؛ يجب أن تكون العقود ملفات PDF أو DOCX نصية."),
    "help.files_title": _s("Files", "الملفات"),
    "help.privacy_title": _s("Your documents", "مستنداتك"),
    "help.privacy_body": _s("Uploaded files are processed for your request only and are not stored.",
                            "تُعالَج الملفات المرفوعة لطلبك فقط ولا يتم تخزينها."),
    "settings.title": _s("Settings", "الإعدادات"),
    "settings.sub": _s("Interface language, and the technical settings used during development.",
                       "لغة الواجهة، والإعدادات التقنية المستخدمة أثناء التطوير."),
    "settings.language": _s("Interface language", "لغة الواجهة"),
    "settings.developer": _s("Developer settings", "إعدادات المطوّر"),
    "settings.developer_note": _s("These are only needed when running Sanad locally.",
                                  "تُستخدم هذه الإعدادات عند تشغيل سند محلياً فقط."),
    "settings.service_address": _s("Service address", "عنوان الخدمة"),
    "settings.connected": _s("Service reachable", "الخدمة متاحة"),
    "settings.disconnected": _s("Service unreachable", "الخدمة غير متاحة"),
    "settings.capabilities": _s("Capabilities reported by the service", "الإمكانات التي تعلنها الخدمة"),
    "settings.answers_on": _s("Written answers enabled", "الإجابات المكتوبة مفعّلة"),
    "settings.answers_off": _s("Written answers disabled — questions return the regulations only",
                               "الإجابات المكتوبة غير مفعّلة — تُعاد المواد النظامية فقط"),
    "settings.salary_on": _s("Salary benchmarking enabled", "مقارنة الرواتب مفعّلة"),
    "settings.salary_off": _s("Salary benchmarking disabled", "مقارنة الرواتب غير مفعّلة"),
    "settings.show_raw": _s("Show the raw service response on result pages",
                            "إظهار استجابة الخدمة الخام في صفحات النتائج"),
    "settings.raw": _s("Raw service response", "استجابة الخدمة الخام"),
}
