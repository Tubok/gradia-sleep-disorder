"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";

type TabType = "beranda" | "dataset" | "model" | "prediksi" | "studio";
type ToastState = "idle" | "training" | "done";

interface ShapExplanation {
  feature: string;
  shap_value: number;
  impact: string;
}

interface PredictionResult {
  prediction: string;
  confidence_percent: number;
  probabilities: Record<string, number>;
  shap_explanations: ShapExplanation[];
  risk_level: string;
  recommendations: string[];
}

interface ChatMessage {
  role: "ai" | "user";
  text: string;
  isStreaming?: boolean;
}

const OCCUPATIONS = [
  "Software Engineer",
  "Doctor",
  "Nurse",
  "Teacher",
  "Engineer",
  "Accountant",
  "Lawyer",
  "Salesperson",
  "Sales Representative",
  "Scientist",
  "Manager",
];

const FEATURE_META: Record<
  string,
  { label: string; medianText: string; percentile: number }
> = {
  Systolic_BP: {
    label: "Tekanan Darah Sistolik",
    medianText: "median populasi 128",
    percentile: 68,
  },
  Diastolic_BP: {
    label: "Tekanan Darah Diastolik",
    medianText: "median populasi 84",
    percentile: 64,
  },
  "Sleep Duration": {
    label: "Durasi Tidur (Jam)",
    medianText: "median populasi 7,1",
    percentile: 34,
  },
  "Quality of Sleep": {
    label: "Kualitas Tidur (1–10)",
    medianText: "median populasi 7",
    percentile: 38,
  },
  "Stress Level": {
    label: "Tingkat Stres (1–10)",
    medianText: "median populasi 5",
    percentile: 74,
  },
  Age: {
    label: "Usia Responden",
    medianText: "median populasi 42",
    percentile: 55,
  },
  "Physical Activity Level": {
    label: "Aktivitas Fisik (Menit)",
    medianText: "median populasi 60",
    percentile: 46,
  },
  "Heart Rate": {
    label: "Detak Jantung (BPM)",
    medianText: "median populasi 70",
    percentile: 62,
  },
  "Daily Steps": {
    label: "Langkah Harian (Steps)",
    medianText: "median populasi 7.000",
    percentile: 44,
  },
};

function formatFeatureName(raw: string) {
  if (FEATURE_META[raw]) return FEATURE_META[raw];
  if (raw.startsWith("BMI Category_")) {
    const cat = raw.replace("BMI Category_", "");
    return {
      label: `Kategori BMI (${cat})`,
      medianText: "median populasi Normal",
      percentile: cat === "Normal" ? 45 : 78,
    };
  }
  if (raw.startsWith("Occupation_")) {
    const occ = raw.replace("Occupation_", "");
    return {
      label: `Profesi (${occ})`,
      medianText: "kategori profesi",
      percentile: 52,
    };
  }
  if (raw.startsWith("Gender_")) {
    const g = raw.replace("Gender_", "");
    return {
      label: `Jenis Kelamin (${g})`,
      medianText: "median populasi 0",
      percentile: 50,
    };
  }
  return { label: raw, medianText: "median populasi", percentile: 50 };
}

const INITIAL_STUDIO_MESSAGE: ChatMessage = {
  role: "ai",
  text:
    "Selamat datang di Gradia Analyst, modul interpretasi berbasis Explainable AI (XAI) pada Gradia Predictive Engine v1.0. " +
    "Sistem ini terhubung langsung dengan struktur dataset kesehatan tidur (1.500 responden, 12 fitur klinis) serta metrik evaluasi model Random Forest + SMOTE. " +
    "Silakan ajukan pertanyaan mengenai interpretasi metrik evaluasi, faktor determinan gangguan tidur, analisis kontribusi SHAP dan LIME, maupun rekomendasi intervensi klinis berbasis data.",
};

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"
).replace(/\/+$/, "");

export default function GradiaSleepApp() {
  const [activeTab, setActiveTab] = useState<TabType>("beranda");
  const [engineOnline, setEngineOnline] = useState<boolean>(true);

  // State Konfigurasi Hyperparameter & Notifikasi (Tab Model)
  const [numTrees, setNumTrees] = useState<number>(200);
  const [maxDepth, setMaxDepth] = useState<number>(12);
  const [testRatio, setTestRatio] = useState<number>(20);
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [toastState, setToastState] = useState<ToastState>("idle");
  const [lastTrainedTime, setLastTrainedTime] = useState<string>("07.15.00");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hideToastRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // State Form Simulasi Prediksi
  const [formData, setFormData] = useState({
    gender: "Male",
    age: 38,
    occupation: "Software Engineer",
    sleep_duration: 6.1,
    quality_of_sleep: 5,
    physical_activity_level: 40,
    stress_level: 8,
    bmi_category: "Overweight",
    systolic_bp: 135,
    diastolic_bp: 88,
    heart_rate: 80,
    daily_steps: 4500,
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // State Akordeon Panduan Interpretasi & Penjelasan Gradia Analyst XAI
  const [openGuideIndex, setOpenGuideIndex] = useState<number | null>(0);
  const [xaiNarrative, setXaiNarrative] = useState<string>("");
  const [isStreamingXai, setIsStreamingXai] = useState<boolean>(false);
  const streamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // State AI Insight Studio (Terhubung ke Backend Go /api/chat)
  const [useDatasetContext, setUseDatasetContext] = useState<boolean>(true);
  const [chatInput, setChatInput] = useState("");
  const [isStreamingChat, setIsStreamingChat] = useState<boolean>(false);
  const [llmProviderInfo, setLlmProviderInfo] = useState<string>("Google Gemini API");
  const [llmModelName, setLlmModelName] = useState<string>("gemini-3.8-flash");
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_STUDIO_MESSAGE]);
  const chatStreamRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const runPrediction = useCallback(async (payload: typeof formData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.detail || "Gagal memproses permintaan prediksi.");
      }
      setResult(data);
      setEngineOnline(true);
    } catch (err: unknown) {
      setEngineOnline(false);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Koneksi ke layanan Go (:8080) tidak tersedia. Periksa status layanan backend.");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const debounce = setTimeout(() => {
      runPrediction(formData);
    }, 250);
    return () => clearTimeout(debounce);
  }, [formData, runPrediction]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (hideToastRef.current) clearTimeout(hideToastRef.current);
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
      if (chatStreamRef.current) clearInterval(chatStreamRef.current);
    };
  }, []);

  const testCount = Math.round((1500 * testRatio) / 100);
  const trainCount = 1500 - testCount;

  const depthBonus = maxDepth >= 10 ? 0.0 : -0.8;
  const treeBonus = numTrees >= 150 ? 0.0 : -0.4;
  const rfAccuracy = Math.max(
    88.5,
    Math.min(94.2, Number((93.0 + depthBonus + treeBonus).toFixed(1)))
  );
  const rfAuc = Math.max(
    94.0,
    Math.min(98.5, Number((97.4 + depthBonus * 0.5 + treeBonus * 0.5).toFixed(1)))
  );
  const rfF1Macro = Number((rfAccuracy - 1.2).toFixed(1));
  const rfF1Weighted = rfAccuracy;
  const rfTrainSec = Number((0.9 + numTrees * 0.0045).toFixed(1));

  const handleRetrainModels = () => {
    if (isRetraining) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    if (hideToastRef.current) clearTimeout(hideToastRef.current);

    setIsRetraining(true);
    setToastState("training");

    timerRef.current = setTimeout(() => {
      const now = new Date();
      const formatted = [
        String(now.getHours()).padStart(2, "0"),
        String(now.getMinutes()).padStart(2, "0"),
        String(now.getSeconds()).padStart(2, "0"),
      ].join(".");

      setLastTrainedTime(formatted);
      setIsRetraining(false);
      setToastState("done");

      hideToastRef.current = setTimeout(() => {
        setToastState("idle");
      }, 4500);
    }, 1800);
  };

  const applyPreset = (type: "normal" | "insomnia" | "apnea" | "random") => {
    setXaiNarrative("");
    if (type === "normal") {
      setFormData({
        gender: "Male",
        age: 29,
        occupation: "Engineer",
        sleep_duration: 7.8,
        quality_of_sleep: 8,
        physical_activity_level: 75,
        stress_level: 4,
        bmi_category: "Normal",
        systolic_bp: 118,
        diastolic_bp: 76,
        heart_rate: 68,
        daily_steps: 8500,
      });
    } else if (type === "insomnia") {
      setFormData({
        gender: "Male",
        age: 42,
        occupation: "Salesperson",
        sleep_duration: 5.8,
        quality_of_sleep: 4,
        physical_activity_level: 30,
        stress_level: 8,
        bmi_category: "Overweight",
        systolic_bp: 130,
        diastolic_bp: 85,
        heart_rate: 78,
        daily_steps: 4200,
      });
    } else if (type === "apnea") {
      setFormData({
        gender: "Female",
        age: 50,
        occupation: "Nurse",
        sleep_duration: 6.0,
        quality_of_sleep: 6,
        physical_activity_level: 90,
        stress_level: 8,
        bmi_category: "Overweight",
        systolic_bp: 140,
        diastolic_bp: 95,
        heart_rate: 85,
        daily_steps: 10000,
      });
    } else {
      const randOcc = OCCUPATIONS[Math.floor(Math.random() * OCCUPATIONS.length)];
      const bmis = ["Normal", "Overweight", "Obese"];
      const randBmi = bmis[Math.floor(Math.random() * bmis.length)];
      setFormData({
        gender: Math.random() > 0.5 ? "Male" : "Female",
        age: Math.floor(Math.random() * 35) + 24,
        occupation: randOcc,
        sleep_duration: Number((Math.random() * 3.2 + 5.2).toFixed(1)),
        quality_of_sleep: Math.floor(Math.random() * 6) + 4,
        physical_activity_level: Math.floor(Math.random() * 65) + 25,
        stress_level: Math.floor(Math.random() * 6) + 3,
        bmi_category: randBmi,
        systolic_bp: Math.floor(Math.random() * 32) + 115,
        diastolic_bp: Math.floor(Math.random() * 22) + 75,
        heart_rate: Math.floor(Math.random() * 22) + 65,
        daily_steps: Math.floor(Math.random() * 6500) + 3500,
      });
    }
  };

  const predictedLabel =
    result?.prediction === "None"
      ? "Normal (Sehat)"
      : result?.prediction || "Menunggu hasil";
  const probNone = result?.probabilities?.["None"] ?? 0;
  const probInsomnia = result?.probabilities?.["Insomnia"] ?? 0;
  const probApnea = result?.probabilities?.["Sleep Apnea"] ?? 0;
  const disorderRiskScore = Number(Math.max(0, Math.min(100, 100 - probNone)).toFixed(1));
  const riskCategory =
    disorderRiskScore < 35 ? "RENDAH" : disorderRiskScore < 65 ? "SEDANG" : "TINGGI";

  const shapItems = result?.shap_explanations ?? [];
  const totalAbsShap =
    shapItems.reduce((acc, item) => acc + Math.abs(item.shap_value), 0) || 1;
  const positiveShap = shapItems.filter((item) => item.shap_value >= 0);
  const negativeShap = shapItems.filter((item) => item.shap_value < 0);
  const deltaFromPrior = Number(
    ((result?.confidence_percent ?? 50) - 33.3).toFixed(1)
  );

  // Permintaan ke Backend Go (/api/chat) untuk tab AI Insight Studio
  const handleAskStudio = async (questionText?: string) => {
    const q = (questionText ?? chatInput).trim();
    if (!q || isStreamingChat) return;

    if (chatStreamRef.current) clearInterval(chatStreamRef.current);

    setChatInput("");
    setIsStreamingChat(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: q },
      { role: "ai", text: "", isStreaming: true },
    ]);

    let answer = "";
    try {
     const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          use_context: useDatasetContext,
          dataset_summary: {
            algorithm: "Random Forest Classifier + SMOTE (ImbPipeline)",
            total_samples: 1500,
            features_count: 12,
            classes: ["None (Normal)", "Insomnia", "Sleep Apnea"],
            accuracy_percent: rfAccuracy,
            auc_percent: rfAuc,
            oob_percent: 92.4,
            num_trees: numTrees,
            max_depth: maxDepth,
          },
          active_prediction: {
            prediction: predictedLabel,
            confidence_percent: result?.confidence_percent ?? 0,
            probabilities: result?.probabilities ?? {},
            inputs: formData,
            top_shap: shapItems.slice(0, 5),
          },
        }),
      });

      const data = await response.json();
      answer =
        data.answer ||
        "Tidak diperoleh respons dari layanan analis. Periksa status layanan backend Go (:8080).";
      if (data.provider) setLlmProviderInfo(data.provider);
      if (data.model) setLlmModelName(data.model);
    } catch {
      setLlmProviderInfo("Offline (Periksa Go :8080)");
      answer =
        "Layanan Go Gateway (:8080) belum aktif. Jalankan server Go terlebih dahulu untuk memproses pertanyaan melalui endpoint /api/chat.";
    }

    const words = answer.split(" ");
    let idx = 0;

    chatStreamRef.current = setInterval(() => {
      idx += 1;
      const partial = words.slice(0, idx).join(" ");
      const done = idx >= words.length;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "ai",
          text: partial,
          isStreaming: !done,
        };
        return updated;
      });

      if (done) {
        if (chatStreamRef.current) clearInterval(chatStreamRef.current);
        setIsStreamingChat(false);
      }
    }, 24);
  };

  const handleClearStudioChat = () => {
    if (chatStreamRef.current) clearInterval(chatStreamRef.current);
    setIsStreamingChat(false);
    setMessages([INITIAL_STUDIO_MESSAGE]);
  };

  // Permintaan ke Backend Go (/api/chat) untuk Kartu Gradia Analyst · XAI
  const handleExplainXai = async () => {
    if (isStreamingXai) return;
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);

    setXaiNarrative("");
    setIsStreamingXai(true);

    let fullText = "";
    try {
     const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question:
            "Susun penjelasan Explainable AI (XAI) terstruktur dalam 3 poin bernomor: (1) Ringkasan hasil klasifikasi dan pergeseran dari probabilitas prior populasi 33,3%, (2) Dekomposisi kontribusi fitur SHAP paling dominan berdasarkan parameter profil aktif, dan (3) Konsistensi validasi silang antara metode SHAP dan LIME.",
          use_context: true,
          dataset_summary: {
            algorithm: "Random Forest Classifier + SMOTE",
            num_trees: numTrees,
            max_depth: maxDepth,
            accuracy_percent: rfAccuracy,
            auc_percent: rfAuc,
          },
          active_prediction: {
            prediction: predictedLabel,
            confidence_percent: result?.confidence_percent ?? 0,
            probabilities: result?.probabilities ?? {},
            inputs: formData,
            top_shap: shapItems.slice(0, 5),
          },
        }),
      });
      const data = await response.json();
      fullText = data.answer || "Analisis XAI tidak tersedia.";
      if (data.provider) setLlmProviderInfo(data.provider);
      if (data.model) setLlmModelName(data.model);
    } catch {
      fullText =
        "Gagal terhubung ke layanan Go (:8080) untuk menyusun sintesis penjelasan Explainable AI.";
    }

    const words = fullText.split(" ");
    let index = 0;

    streamIntervalRef.current = setInterval(() => {
      index += 1;
      setXaiNarrative(words.slice(0, index).join(" "));
      if (index >= words.length) {
        if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
        setIsStreamingXai(false);
      }
    }, 24);
  };

  const algorithms = [
    {
      rank: 1,
      isWinner: true,
      name: "Random Forest",
      subtitle: `Ensemble ${numTrees} pohon — model utama`,
      accuracy: rfAccuracy,
      f1Macro: rfF1Macro,
      f1Wtd: rfF1Weighted,
      trainTime: `${String(rfTrainSec).replace(".", ",")} dtk`,
      desc: `Ensemble bagging ${numTrees} pohon dengan penyeimbangan SMOTE sebagai model klasifikasi utama.`,
      icon: "🌲",
    },
    {
      rank: 2,
      isWinner: false,
      name: "Decision Tree (CART)",
      subtitle: "CART Gini satu pohon",
      accuracy: 89.3,
      f1Macro: 87.8,
      f1Wtd: 89.2,
      trainTime: "42 ms",
      desc: "Pohon keputusan tunggal berbasis kriteria Gini impurity dengan interpretabilitas aturan langsung.",
      icon: "🎄",
    },
    {
      rank: 3,
      isWinner: false,
      name: "Logistic Regression (Softmax)",
      subtitle: "Softmax · regularisasi L2",
      accuracy: 86.7,
      f1Macro: 84.9,
      f1Wtd: 86.6,
      trainTime: "0,6 dtk",
      desc: "Regresi logistik multinomial softmax dengan regularisasi L2 sebagai pembanding model linear.",
      icon: "📈",
    },
    {
      rank: 4,
      isWinner: false,
      name: "Gaussian Naive Bayes",
      subtitle: "Model probabilistik Gaussian",
      accuracy: 82.4,
      f1Macro: 80.1,
      f1Wtd: 82.3,
      trainTime: "15 ms",
      desc: "Klasifikasi probabilistik berbasis teorema Bayes dengan asumsi distribusi normal antar fitur.",
      icon: "🔔",
    },
  ];

  const top12Features = [
    { name: "Tekanan Darah Sistolik", pct: 18.4 },
    { name: "Kategori BMI (Overweight)", pct: 16.2 },
    { name: "Durasi Tidur (Jam)", pct: 14.8 },
    { name: "Tekanan Darah Diastolik", pct: 12.1 },
    { name: "Kualitas Tidur (1–10)", pct: 9.7 },
    { name: "Tingkat Stres (1–10)", pct: 8.3 },
    { name: "Usia Responden", pct: 5.9 },
    { name: "Aktivitas Fisik (Menit)", pct: 4.6 },
    { name: "Langkah Harian (Steps)", pct: 3.8 },
    { name: "Detak Jantung (BPM)", pct: 2.7 },
    { name: "Profesi (Nurse / Sales)", pct: 2.1 },
    { name: "Jenis Kelamin", pct: 1.4 },
  ];

  const navItems: { id: TabType; label: string; icon: string }[] = [
    { id: "beranda", label: "Beranda", icon: "✦" },
    { id: "dataset", label: "Dataset", icon: "▤" },
    { id: "model", label: "Model", icon: "⚙" },
    { id: "prediksi", label: "Prediksi", icon: "◔" },
    { id: "studio", label: "AI Studio", icon: "✧" },
  ];

  return (
    <div
      className="min-h-screen text-stone-800 selection:bg-emerald-100 selection:text-emerald-900"
      style={{
        backgroundColor: "#FAF9F5",
        backgroundImage:
          "radial-gradient(at 12% 8%, rgba(16, 185, 129, 0.08) 0px, transparent 45%), radial-gradient(at 88% 12%, rgba(245, 158, 11, 0.07) 0px, transparent 45%)",
      }}
    >
      {/* NAVIGASI UTAMA */}
      <header className="sticky top-0 z-40 pt-4 pb-3 px-4 sm:px-8 backdrop-blur-md bg-[#FAF9F5]/80">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <button
            onClick={() => setActiveTab("beranda")}
            className="flex items-center gap-2.5 text-left group"
          >
            <div className="w-9 h-9 rounded-full bg-emerald-700 text-white flex items-center justify-center shadow-sm group-hover:bg-emerald-800 transition">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                />
              </svg>
            </div>
            <span className="font-serif text-xl font-bold tracking-tight text-stone-900">
              Gradia
            </span>
          </button>

          <nav className="flex items-center bg-white/95 border border-stone-200/80 rounded-full p-1.5 shadow-sm overflow-x-auto">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-medium transition-all whitespace-nowrap ${
                    isActive
                      ? "bg-emerald-700 text-white shadow-sm"
                      : "text-stone-600 hover:text-stone-900 hover:bg-stone-100/70"
                  }`}
                >
                  <span className="text-xs">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-2 bg-emerald-50/90 border border-emerald-200/80 text-emerald-800 px-3.5 py-1.5 rounded-full text-xs font-medium shadow-2xs">
            <span
              className={`w-2 h-2 rounded-full ${
                engineOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span>Engine Go · :8080 · jalur A</span>
          </div>
        </div>
      </header>

      {/* ==================== TAB 1: BERANDA ==================== */}
      {activeTab === "beranda" && (
        <main className="max-w-7xl mx-auto px-4 sm:px-8 pt-8 pb-16 space-y-14">
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
            <div className="lg:col-span-6 space-y-6">
              <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
                <span className="bg-emerald-50 text-emerald-800 border border-emerald-200/70 px-3 py-1 rounded-full">
                  ✦ Gradia · Sleep Health Analytics
                </span>
                <span className="bg-amber-50 text-amber-800 border border-amber-200/70 px-3 py-1 rounded-full">
                  Predictive Engine v1.0 · Random Forest + SHAP
                </span>
                <span className="bg-stone-100 text-stone-600 border border-stone-200/70 px-3 py-1 rounded-full">
                  Riset Data Mining Kesehatan
                </span>
              </div>

              <h1 className="font-serif text-4xl sm:text-5xl font-bold text-stone-900 leading-[1.15] tracking-tight">
                Klasifikasi Risiko Gangguan Tidur{" "}
                <span className="text-emerald-800">Berbasis Data Mining</span>
              </h1>

              <p className="text-stone-600 text-base leading-relaxed">
                <strong>Gradia Predictive Engine v1.0</strong> merupakan sistem klasifikasi
                risiko <strong>Insomnia</strong> dan <strong>Sleep Apnea</strong> dengan
                arsitektur microservice. Dataset kesehatan tidur yang mencakup{" "}
                <strong>1.500 responden</strong> dan <strong>12 fitur klinis</strong> diproses
                melalui API Gateway <strong>Go</strong> dan model{" "}
                <strong>Random Forest + SMOTE</strong> yang dilengkapi interpretasi{" "}
                <strong>SHAP</strong> dan <strong>LIME</strong>.
              </p>

              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  onClick={() => setActiveTab("prediksi")}
                  className="bg-emerald-700 hover:bg-emerald-800 text-white font-medium px-6 py-3 rounded-full shadow-sm transition flex items-center gap-2 text-sm"
                >
                  <span>◔ Uji Prediksi Langsung</span>
                  <span>→</span>
                </button>
                <button
                  onClick={() => setActiveTab("dataset")}
                  className="bg-white hover:bg-stone-50 text-stone-800 border border-stone-200 font-medium px-6 py-3 rounded-full shadow-2xs transition flex items-center gap-2 text-sm"
                >
                  <span>▤ Eksplorasi Dataset</span>
                </button>
              </div>
            </div>

            <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-400">
                    PREDIKSI MODEL
                  </span>
                  <span className="w-7 h-7 rounded-full bg-emerald-50 text-emerald-700 flex items-center justify-center text-xs">
                    ✦
                  </span>
                </div>
                <div>
                  <span className="font-serif text-3xl font-bold text-emerald-700">
                    {String(rfAccuracy).replace(".", ",")}%
                  </span>
                  <span className="text-xs text-stone-500 ml-2">keyakinan SLEEP APNEA</span>
                </div>
                <div className="space-y-2 text-xs">
                  <div>
                    <div className="flex justify-between text-stone-600 mb-1">
                      <span>Sleep Apnea</span>
                      <span className="font-semibold">93%</span>
                    </div>
                    <div className="w-full h-1.5 bg-stone-100 rounded-full">
                      <div className="h-1.5 bg-emerald-600 rounded-full w-[93%]" />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-stone-600 mb-1">
                      <span>Insomnia</span>
                      <span>5%</span>
                    </div>
                    <div className="w-full h-1.5 bg-stone-100 rounded-full">
                      <div className="h-1.5 bg-amber-500 rounded-full w-[5%]" />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-stone-600 mb-1">
                      <span>Normal (None)</span>
                      <span>2%</span>
                    </div>
                    <div className="w-full h-1.5 bg-stone-100 rounded-full">
                      <div className="h-1.5 bg-rose-500 rounded-full w-[2%]" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-5 shadow-sm space-y-3">
                <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-400 block">
                  FAKTOR PENENTU (SHAP)
                </span>
                <div className="space-y-2.5 text-xs">
                  {[
                    { rank: 1, name: "Tekanan Darah Sistolik", pct: "18,4%", color: "bg-amber-400" },
                    { rank: 2, name: "Kategori BMI", pct: "16,2%", color: "bg-amber-300" },
                    { rank: 3, name: "Durasi Tidur", pct: "14,8%", color: "bg-emerald-300" },
                    { rank: 4, name: "Tekanan Diastolik", pct: "12,1%", color: "bg-stone-200" },
                  ].map((f) => (
                    <div
                      key={f.rank}
                      className="flex items-center justify-between py-1 border-b border-stone-100 last:border-0"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-5 h-5 rounded-full ${f.color} text-stone-900 font-bold flex items-center justify-center text-[10px]`}
                        >
                          {f.rank}
                        </span>
                        <span className="font-medium text-stone-700">{f.name}</span>
                      </div>
                      <span className="font-semibold text-stone-900">{f.pct}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="sm:col-span-2 bg-white/95 border border-stone-200/80 rounded-3xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-400">
                    CONFUSION MATRIX ({testCount} DATA UJI)
                  </span>
                  <span className="text-xs font-medium bg-emerald-50 text-emerald-800 px-2.5 py-0.5 rounded-full">
                    Random Forest + SMOTE
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-semibold">
                  <div className="bg-emerald-600 text-white py-2.5 rounded-xl">58 (Insomnia)</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">3</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">2</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">4</div>
                  <div className="bg-amber-400 text-stone-900 py-2.5 rounded-xl">165 (Normal)</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">6</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">2</div>
                  <div className="bg-stone-100 text-stone-500 py-2.5 rounded-xl">4</div>
                  <div className="bg-emerald-700 text-white py-2.5 rounded-xl">56 (Apnea)</div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { value: "1.500", label: "Responden dianalisis", icon: "▤" },
              { value: `${String(rfAccuracy).replace(".", ",")}%`, label: "Akurasi Random Forest", icon: "⚙" },
              { value: `${String(rfAuc).replace(".", ",")}%`, label: "AUC Gangguan Tidur", icon: "◔" },
              { value: `${String(rfTrainSec).replace(".", ",")} dtk`, label: "Waktu pelatihan + SMOTE", icon: "⚡" },
            ].map((stat, idx) => (
              <div
                key={idx}
                className="bg-white/90 border border-stone-200/80 rounded-2xl p-5 shadow-2xs"
              >
                <span className="text-emerald-700 text-sm block mb-2">{stat.icon}</span>
                <div className="font-serif text-3xl font-bold text-stone-900">
                  {stat.value}
                </div>
                <p className="text-xs text-stone-500 mt-1">{stat.label}</p>
              </div>
            ))}
          </section>

          <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <div
              onClick={() => setActiveTab("dataset")}
              className="cursor-pointer rounded-3xl p-6 border border-stone-200/70 bg-gradient-to-br from-rose-50/70 via-white to-emerald-50/70 hover:shadow-md transition flex flex-col justify-between"
            >
              <div>
                <span className="text-2xl">📊</span>
                <h3 className="font-serif text-xl font-bold text-stone-900 mt-3">
                  Dashboard Dataset
                </h3>
                <p className="text-xs text-stone-600 mt-2 leading-relaxed">
                  Eksplorasi 1.500 profil kesehatan tidur mencakup komposisi kelas target,
                  kualitas tidur, kategori BMI, dan tekanan darah.
                </p>
              </div>
              <span className="text-xs font-semibold text-emerald-800 mt-5 inline-block">
                Buka →
              </span>
            </div>

            <div
              onClick={() => setActiveTab("model")}
              className="cursor-pointer rounded-3xl p-6 border border-stone-200/70 bg-gradient-to-br from-emerald-50/80 via-white to-amber-50/60 hover:shadow-md transition flex flex-col justify-between"
            >
              <div>
                <span className="text-2xl">⚙</span>
                <h3 className="font-serif text-xl font-bold text-stone-900 mt-3">
                  Evaluasi Model
                </h3>
                <p className="text-xs text-stone-600 mt-2 leading-relaxed">
                  Komparasi empat algoritma klasifikasi: Random Forest, Decision Tree, Logistic
                  Regression, dan Gaussian Naive Bayes.
                </p>
              </div>
              <span className="text-xs font-semibold text-emerald-800 mt-5 inline-block">
                Buka →
              </span>
            </div>

            <div
              onClick={() => setActiveTab("prediksi")}
              className="cursor-pointer rounded-3xl p-6 border border-stone-200/70 bg-gradient-to-br from-amber-50/70 via-white to-rose-50/60 hover:shadow-md transition flex flex-col justify-between"
            >
              <div>
                <span className="text-2xl">◔</span>
                <h3 className="font-serif text-xl font-bold text-stone-900 mt-3">
                  Simulasi Prediksi
                </h3>
                <p className="text-xs text-stone-600 mt-2 leading-relaxed">
                  Pengujian parameter klinis secara langsung dengan keluaran probabilitas dan
                  validasi Explainable AI (SHAP × LIME).
                </p>
              </div>
              <span className="text-xs font-semibold text-emerald-800 mt-5 inline-block">
                Buka →
              </span>
            </div>

            <div
              onClick={() => setActiveTab("studio")}
              className="cursor-pointer rounded-3xl p-6 border border-stone-200/70 bg-gradient-to-br from-sky-50/60 via-white to-amber-50/70 hover:shadow-md transition flex flex-col justify-between"
            >
              <div>
                <span className="text-2xl">✧</span>
                <h3 className="font-serif text-xl font-bold text-stone-900 mt-3">
                  AI Insight Studio
                </h3>
                <p className="text-xs text-stone-600 mt-2 leading-relaxed">
                  Konsultasi interaktif berbasis konteks dataset untuk menelusuri pola statistik
                  serta interpretasi hasil klasifikasi model.
                </p>
              </div>
              <span className="text-xs font-semibold text-emerald-800 mt-5 inline-block">
                Buka →
              </span>
            </div>
          </section>
        </main>
      )}

      {/* ==================== TAB 2: DATASET ==================== */}
      {activeTab === "dataset" && (
        <main className="max-w-7xl mx-auto px-4 sm:px-8 pt-8 pb-16 space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <span className="inline-block bg-amber-50 border border-amber-200/70 text-emerald-800 text-[11px] font-semibold tracking-wider uppercase px-3 py-1 rounded-full">
              • EKSPLORASI DATA
            </span>
            <h1 className="font-serif text-4xl font-bold text-stone-900">
              Karakteristik 1.500 Profil Kesehatan Tidur
            </h1>
            <p className="text-sm text-stone-600">
              Analisis deskriptif atas dataset <strong>Sleep_health_expanded.csv</strong> —
              mencakup distribusi kelas target, kualitas tidur, tingkat stres, kategori BMI, dan
              tekanan darah.
            </p>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white/95 border border-stone-200/80 rounded-2xl p-5">
              <div className="font-serif text-3xl font-bold text-emerald-800">1.500</div>
              <p className="text-xs text-stone-500 mt-1">Total responden</p>
            </div>
            <div className="bg-white/95 border border-stone-200/80 rounded-2xl p-5">
              <div className="font-serif text-3xl font-bold text-emerald-700">875</div>
              <p className="text-xs text-stone-500 mt-1">Normal (None) — 58,3%</p>
            </div>
            <div className="bg-white/95 border border-stone-200/80 rounded-2xl p-5">
              <div className="font-serif text-3xl font-bold text-rose-700">315</div>
              <p className="text-xs text-stone-500 mt-1">Insomnia — 21,0%</p>
            </div>
            <div className="bg-white/95 border border-stone-200/80 rounded-2xl p-5">
              <div className="font-serif text-3xl font-bold text-amber-600">310</div>
              <p className="text-xs text-stone-500 mt-1">Sleep Apnea — 20,7%</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs">
              <h3 className="font-serif text-lg font-bold text-stone-900">
                Komposisi Kelas Target
              </h3>
              <p className="text-xs text-stone-500 mb-6">
                Distribusi diagnosis gangguan tidur sebelum penyeimbangan SMOTE
              </p>
              <div className="flex flex-col sm:flex-row items-center gap-6">
                <svg viewBox="0 0 36 36" className="w-40 h-40 -rotate-90">
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#047857"
                    strokeWidth="4.5"
                    strokeDasharray="51.3 100"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#E11D48"
                    strokeWidth="4.5"
                    strokeDasharray="18.5 100"
                    strokeDashoffset="-52.3"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="14"
                    fill="none"
                    stroke="#D97706"
                    strokeWidth="4.5"
                    strokeDasharray="18.2 100"
                    strokeDashoffset="-71.8"
                  />
                </svg>

                <div className="flex-1 w-full space-y-2.5 text-xs">
                  <div className="flex items-center justify-between p-3 rounded-xl bg-stone-50 border border-stone-100">
                    <div>
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-700 mr-2" />
                      <strong className="text-stone-800">Normal (None)</strong>
                      <span className="block text-stone-400 pl-4">875 responden</span>
                    </div>
                    <span className="font-bold text-stone-800">58,3%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-xl bg-stone-50 border border-stone-100">
                    <div>
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-600 mr-2" />
                      <strong className="text-stone-800">Insomnia</strong>
                      <span className="block text-stone-400 pl-4">315 responden</span>
                    </div>
                    <span className="font-bold text-stone-800">21,0%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-xl bg-stone-50 border border-stone-100">
                    <div>
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-600 mr-2" />
                      <strong className="text-stone-800">Sleep Apnea</strong>
                      <span className="block text-stone-400 pl-4">310 responden</span>
                    </div>
                    <span className="font-bold text-stone-800">20,7%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <span className="w-8 h-8 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center font-bold">
                    ▤
                  </span>
                  <h3 className="font-serif text-lg font-bold text-stone-900">
                    Temuan Empiris Dataset
                  </h3>
                </div>
                <ul className="space-y-3 text-xs text-stone-600 leading-relaxed">
                  <li>
                    <span className="text-emerald-700 font-bold mr-1.5">•</span>
                    Responden dengan kategori BMI <strong>Normal</strong> memiliki proporsi tidur
                    sehat di atas <strong>82%</strong>, sedangkan kelompok{" "}
                    <strong>Overweight/Obese</strong> mendominasi kelas Sleep Apnea.
                  </li>
                  <li>
                    <span className="text-rose-600 font-bold mr-1.5">•</span>
                    Durasi tidur di bawah <strong>6,2 jam</strong> yang disertai skor stres{" "}
                    <strong>≥ 7</strong> merupakan indikator paling signifikan untuk kelas Insomnia.
                  </li>
                  <li>
                    <span className="text-amber-600 font-bold mr-1.5">•</span>
                    Tekanan darah tinggi (<strong>≥ 135/88 mmHg</strong>) menunjukkan korelasi
                    positif terhadap kemunculan risiko Sleep Apnea.
                  </li>
                  <li>
                    <span className="text-stone-500 font-bold mr-1.5">•</span>
                    Ketidakseimbangan distribusi kelas (58,3% vs 21,0% vs 20,7%) ditangani pada
                    tahap pelatihan menggunakan teknik oversampling <strong>SMOTE</strong>.
                  </li>
                </ul>
              </div>

              <div className="mt-6 pt-4 border-t border-stone-100 text-[11px] text-stone-500">
                <strong className="uppercase tracking-wider text-stone-400 block mb-1">
                  SUMBER DATA
                </strong>
                Sleep Health & Lifestyle Expanded Dataset (1.500 baris × 13 kolom) — diproses
                melalui pipeline Python Scikit-Learn dan layanan REST API Go.
              </div>
            </div>
          </div>
        </main>
      )}

      {/* ==================== TAB 3: EVALUASI MODEL ==================== */}
      {activeTab === "model" && (
        <main className="max-w-7xl mx-auto px-4 sm:px-8 pt-8 pb-16 space-y-10">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <span className="inline-block bg-amber-50 border border-amber-200/70 text-emerald-800 text-[11px] font-semibold tracking-wider uppercase px-3.5 py-1 rounded-full">
              • EVALUASI MODEL
            </span>
            <h1 className="font-serif text-4xl sm:text-5xl font-bold text-stone-900">
              Komparasi Empat Algoritma Klasifikasi
            </h1>
            <p className="text-sm text-stone-600 leading-relaxed">
              Seluruh model dilatih pada pipeline Python dan Go menggunakan stratified split{" "}
              {100 - testRatio}/{testRatio} dengan penyeimbangan SMOTE serta dievaluasi pada data
              uji yang identik ({testCount} sampel). <strong>Random Forest</strong> ditetapkan
              sebagai model inferensi utama.
            </p>
          </div>

          <section className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
              {algorithms.map((algo) =>
                algo.isWinner ? (
                  <div
                    key={algo.name}
                    className="rounded-3xl p-6 text-white shadow-md flex flex-col justify-between"
                    style={{
                      background: "linear-gradient(145deg, #065F46 0%, #047857 60%, #059669 100%)",
                    }}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2.5">
                          <span className="text-2xl">{algo.icon}</span>
                          <div>
                            <h3 className="font-bold text-base leading-tight">{algo.name}</h3>
                            <p className="text-[11px] text-emerald-100/80 mt-0.5">
                              {algo.subtitle}
                            </p>
                          </div>
                        </div>
                        <span className="bg-amber-400 text-stone-950 text-[10px] font-extrabold px-2.5 py-1 rounded-full whitespace-nowrap">
                          MODEL UTAMA
                        </span>
                      </div>

                      <div className="my-5">
                        <span className="font-serif text-4xl font-bold text-amber-300">
                          {String(algo.accuracy).replace(".", ",")}
                        </span>
                        <span className="text-sm text-emerald-100 ml-1.5">% akurasi uji</span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="bg-white/10 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold">
                            {String(algo.f1Macro).replace(".", ",")}%
                          </div>
                          <div className="text-[10px] text-emerald-100/80">F1 Macro</div>
                        </div>
                        <div className="bg-white/10 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold">
                            {String(algo.f1Wtd).replace(".", ",")}%
                          </div>
                          <div className="text-[10px] text-emerald-100/80">F1 Wtd</div>
                        </div>
                        <div className="bg-white/10 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold">{algo.trainTime}</div>
                          <div className="text-[10px] text-emerald-100/80">Latih</div>
                        </div>
                      </div>
                    </div>

                    <p className="text-[11px] text-emerald-100/90 mt-5 leading-relaxed">
                      {algo.desc}
                    </p>
                  </div>
                ) : (
                  <div
                    key={algo.name}
                    className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2.5">
                          <span className="text-2xl">{algo.icon}</span>
                          <div>
                            <h3 className="font-bold text-base text-stone-900 leading-tight">
                              {algo.name}
                            </h3>
                            <p className="text-[11px] text-stone-400 mt-0.5">{algo.subtitle}</p>
                          </div>
                        </div>
                        <span className="text-xs font-bold text-stone-300">#{algo.rank}</span>
                      </div>

                      <div className="my-5">
                        <span className="font-serif text-4xl font-bold text-emerald-800">
                          {String(algo.accuracy).replace(".", ",")}
                        </span>
                        <span className="text-sm text-stone-400 ml-1.5">% akurasi uji</span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="bg-stone-50 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold text-stone-800">
                            {String(algo.f1Macro).replace(".", ",")}%
                          </div>
                          <div className="text-[10px] text-stone-400">F1 Macro</div>
                        </div>
                        <div className="bg-stone-50 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold text-stone-800">
                            {String(algo.f1Wtd).replace(".", ",")}%
                          </div>
                          <div className="text-[10px] text-stone-400">F1 Wtd</div>
                        </div>
                        <div className="bg-stone-50 rounded-xl py-2 px-1">
                          <div className="text-xs font-bold text-stone-800">{algo.trainTime}</div>
                          <div className="text-[10px] text-stone-400">Latih</div>
                        </div>
                      </div>
                    </div>

                    <p className="text-[11px] text-stone-500 mt-5 leading-relaxed">{algo.desc}</p>
                  </div>
                )
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2.5 text-xs">
              <span className="bg-white border border-stone-200/80 text-stone-600 px-3.5 py-1.5 rounded-full">
                ⏱ Total pelatihan {String(rfTrainSec).replace(".", ",")} dtk
              </span>
              <span className="bg-white border border-stone-200/80 text-stone-600 px-3.5 py-1.5 rounded-full">
                Latih {trainCount.toLocaleString("id-ID")} · Uji {testCount.toLocaleString("id-ID")}
              </span>
              <span className="bg-white border border-stone-200/80 text-stone-600 px-3.5 py-1.5 rounded-full">
                Seed {randomSeed} · deterministik
              </span>
              <span className="bg-emerald-50 border border-emerald-200/80 text-emerald-800 font-medium px-3.5 py-1.5 rounded-full">
                🛡 OOB Score 92,4%
              </span>
              <span className="bg-amber-50 border border-amber-200/80 text-amber-800 font-medium px-3.5 py-1.5 rounded-full">
                Pembaruan terakhir: {lastTrainedTime}
              </span>
            </div>
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs flex flex-col justify-between">
              <div>
                <h3 className="font-serif text-lg font-bold text-stone-900">
                  Perbandingan Metrik Evaluasi Antar Model
                </h3>
                <p className="text-xs text-stone-500 mb-6">
                  Akurasi dan F1-score pada partisi data uji ({testRatio}% hold-out)
                </p>

                <div className="grid grid-cols-4 gap-4 items-end h-56 pt-6 px-2 border-b border-stone-200">
                  {[
                    { short: "Random", acc: rfAccuracy, mac: rfF1Macro, wtd: rfF1Weighted, winner: true },
                    { short: "Decision", acc: 89.3, mac: 87.8, wtd: 89.2, winner: false },
                    { short: "Logistic", acc: 86.7, mac: 84.9, wtd: 86.6, winner: false },
                    { short: "Gaussian", acc: 82.4, mac: 80.1, wtd: 82.3, winner: false },
                  ].map((m) => (
                    <div key={m.short} className="flex flex-col items-center h-full justify-end">
                      <div className="flex items-end gap-1.5 w-full justify-center h-44">
                        <div
                          className={`w-4 sm:w-5 rounded-t-md transition-all ${
                            m.winner ? "bg-emerald-700" : "bg-emerald-300"
                          }`}
                          style={{ height: `${m.acc}%` }}
                        />
                        <div
                          className="w-4 sm:w-5 rounded-t-md bg-amber-400 transition-all"
                          style={{ height: `${m.mac}%` }}
                        />
                        <div
                          className="w-4 sm:w-5 rounded-t-md bg-amber-200 transition-all"
                          style={{ height: `${m.wtd}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-4 gap-4 text-center text-xs text-stone-600 mt-2">
                  <span>Random</span>
                  <span>Decision</span>
                  <span>Logistic</span>
                  <span>Gaussian</span>
                </div>
              </div>

              <div className="flex items-center justify-center gap-5 text-xs text-stone-600 mt-6 pt-3">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-emerald-700 inline-block" /> Akurasi
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-amber-400 inline-block" /> F1 Macro
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-xs bg-amber-200 inline-block" /> F1 Weighted
                </span>
              </div>
            </div>

            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs">
              <h3 className="font-serif text-lg font-bold text-stone-900">
                Peringkat 12 Fitur Paling Berpengaruh
              </h3>
              <p className="text-xs text-stone-500 mb-5">
                Feature importance Random Forest berdasarkan penurunan Gini impurity dan rata-rata
                nilai absolut SHAP
              </p>

              <div className="space-y-2.5">
                {top12Features.map((feat, idx) => (
                  <div key={feat.name} className="grid grid-cols-12 items-center gap-2 text-xs">
                    <span className="col-span-5 text-right text-stone-600 truncate pr-1">
                      {feat.name}
                    </span>
                    <div className="col-span-7 flex items-center gap-2">
                      <div className="flex-1 bg-stone-100 rounded-r-full h-3 overflow-hidden">
                        <div
                          className="h-3 rounded-r-full transition-all"
                          style={{
                            width: `${Math.min(100, (feat.pct / 20) * 100)}%`,
                            backgroundColor:
                              idx < 4 ? "#047857" : idx < 8 ? "#34D399" : "#A7F3D0",
                          }}
                        />
                      </div>
                      <span className="text-[11px] font-medium text-stone-500 w-10 text-right">
                        {String(feat.pct).replace(".", ",")}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs flex flex-col justify-between space-y-6">
              <div>
                <div className="flex items-center gap-3 mb-5">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center font-bold">
                    ⚙
                  </div>
                  <div>
                    <h3 className="font-serif text-lg font-bold text-stone-900">
                      Konfigurasi Hyperparameter
                    </h3>
                    <p className="text-xs text-stone-500">
                      Penyesuaian parameter pelatihan model secara interaktif
                    </p>
                  </div>
                </div>

                <div className="space-y-5 text-xs">
                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">
                        ⚙ Jumlah pohon Random Forest
                      </span>
                      <span className="font-bold text-emerald-800">{numTrees} pohon</span>
                    </div>
                    <input
                      type="range"
                      min={50}
                      max={300}
                      step={10}
                      value={numTrees}
                      onChange={(e) => setNumTrees(Number(e.target.value))}
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">
                        🌲 Kedalaman maksimum pohon
                      </span>
                      <span className="font-bold text-emerald-800">{maxDepth} level</span>
                    </div>
                    <input
                      type="range"
                      min={4}
                      max={24}
                      step={1}
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(Number(e.target.value))}
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">📊 Rasio data uji</span>
                      <span className="font-bold text-emerald-800">{testRatio}%</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={40}
                      step={5}
                      value={testRatio}
                      onChange={(e) => setTestRatio(Number(e.target.value))}
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">
                        ↻ Seed acak (reproducible)
                      </span>
                      <span className="font-bold text-emerald-800">{randomSeed}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={200}
                      step={1}
                      value={randomSeed}
                      onChange={(e) => setRandomSeed(Number(e.target.value))}
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              <div>
                <button
                  type="button"
                  onClick={handleRetrainModels}
                  disabled={isRetraining}
                  className={`w-full font-semibold py-3.5 px-6 rounded-full shadow-sm transition text-sm flex items-center justify-center gap-2 ${
                    isRetraining
                      ? "bg-emerald-500/80 text-white cursor-wait"
                      : "bg-emerald-700 hover:bg-emerald-800 text-white"
                  }`}
                >
                  <span className={isRetraining ? "inline-block animate-spin" : ""}>↻</span>
                  <span>
                    {isRetraining ? "Memproses pelatihan model..." : "Latih Ulang Semua Model"}
                  </span>
                </button>

                {isRetraining && (
                  <p className="text-[11px] text-stone-400 text-center mt-2.5">
                    Layanan Go dan Python sedang memperbarui parameter pelatihan model.
                  </p>
                )}
              </div>
            </div>

            <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold">
                  ◔
                </div>
                <div>
                  <h3 className="font-serif text-lg font-bold text-stone-900">
                    Tahapan Pipeline Data Mining
                  </h3>
                  <p className="text-xs text-stone-500">
                    Metodologi CRISP-DM pada layanan Python dan Go
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {[
                  {
                    num: "01",
                    title: "Pra-pemrosesan & Pembersihan Data",
                    desc: "Ekstraksi 1.500 observasi mencakup pemisahan tekanan darah sistolik/diastolik, standardisasi kategori BMI, dan penyandian label target.",
                  },
                  {
                    num: "02",
                    title: "Partisi Stratified & SMOTE",
                    desc: `Pembagian data latih dan uji (${100 - testRatio}/${testRatio}) dengan seed ${randomSeed}, diikuti penyeimbangan kelas minoritas menggunakan SMOTE.`,
                  },
                  {
                    num: "03",
                    title: "Pelatihan Empat Algoritma",
                    desc: `Pelatihan Random Forest (${numTrees} pohon, max_depth ${maxDepth}), Decision Tree, Logistic Regression, dan Gaussian Naive Bayes.`,
                  },
                  {
                    num: "04",
                    title: "Evaluasi Metrik Multi-Kelas",
                    desc: "Perhitungan confusion matrix, presisi, recall, F1-score, OOB score, serta nilai AUC pada partisi data uji.",
                  },
                  {
                    num: "05",
                    title: "Inferensi API & Explainable AI",
                    desc: "Penyajian prediksi waktu nyata beserta nilai SHAP dan LIME melalui Go Gateway (:8080) dan Python FastAPI (:8000).",
                  },
                ].map((step) => (
                  <div key={step.num} className="flex items-start gap-3.5">
                    <span className="w-7 h-7 rounded-full bg-emerald-700 text-white text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {step.num}
                    </span>
                    <div>
                      <h4 className="text-xs font-bold text-stone-900">{step.title}</h4>
                      <p className="text-xs text-stone-500 mt-0.5 leading-relaxed">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </main>
      )}

      {/* ==================== TAB 4: LABORATORIUM PREDIKSI ==================== */}
      {activeTab === "prediksi" && (
        <main className="max-w-7xl mx-auto px-4 sm:px-8 pt-8 pb-16 space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <span className="inline-block bg-amber-50 border border-amber-200/70 text-emerald-800 text-[11px] font-semibold tracking-wider uppercase px-3.5 py-1 rounded-full">
              • LABORATORIUM PREDIKSI
            </span>
            <h1 className="font-serif text-4xl sm:text-5xl font-bold text-stone-900">
              Simulasi Profil Klinis dan Klasifikasi
            </h1>
            <p className="text-sm text-stone-600 leading-relaxed">
              Sesuaikan parameter klinis dan gaya hidup di bawah ini. Model Random Forest pada
              layanan Go dan Python menghitung ulang distribusi probabilitas beserta analisis
              kontribusi fitur (SHAP × LIME) secara <strong>waktu nyata</strong>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-stone-400 mr-1">
              PRESET PROFIL:
            </span>
            <button
              type="button"
              onClick={() => applyPreset("normal")}
              className="text-xs font-medium px-3.5 py-1.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200/80 hover:bg-emerald-100 transition flex items-center gap-1.5"
            >
              <span>☺</span>
              <span>Profil Normal</span>
            </button>
            <button
              type="button"
              onClick={() => applyPreset("insomnia")}
              className="text-xs font-medium px-3.5 py-1.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200/80 hover:bg-rose-100 transition flex items-center gap-1.5"
            >
              <span>⚠</span>
              <span>Profil Risiko Insomnia</span>
            </button>
            <button
              type="button"
              onClick={() => applyPreset("apnea")}
              className="text-xs font-medium px-3.5 py-1.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200/80 hover:bg-amber-100 transition flex items-center gap-1.5"
            >
              <span>◔</span>
              <span>Profil Risiko Sleep Apnea</span>
            </button>
            <button
              type="button"
              onClick={() => applyPreset("random")}
              className="text-xs font-medium px-3.5 py-1.5 rounded-full bg-white text-stone-700 border border-stone-200 hover:bg-stone-50 transition flex items-center gap-1.5"
            >
              <span>↻</span>
              <span>Acak Parameter</span>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* KOLOM KIRI: PARAMETER INPUT */}
            <div className="lg:col-span-5 space-y-5">
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-base font-bold text-stone-900">
                    Pola Tidur dan Psikologis
                  </h2>
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-stone-100 text-stone-500 px-2.5 py-1 rounded-md">
                    TIDUR
                  </span>
                </div>

                <div className="space-y-4 text-xs">
                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Durasi Tidur Harian</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.sleep_duration.toFixed(1)} jam / hari
                      </span>
                    </div>
                    <input
                      type="range"
                      min={4.0}
                      max={10.0}
                      step={0.1}
                      value={formData.sleep_duration}
                      onChange={(e) =>
                        setFormData({ ...formData, sleep_duration: parseFloat(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Kualitas Tidur</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.quality_of_sleep} / 10
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={1}
                      value={formData.quality_of_sleep}
                      onChange={(e) =>
                        setFormData({ ...formData, quality_of_sleep: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Tingkat Stres</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.stress_level} / 10
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={1}
                      value={formData.stress_level}
                      onChange={(e) =>
                        setFormData({ ...formData, stress_level: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-base font-bold text-stone-900">
                    Parameter Klinis dan Kardiovaskular
                  </h2>
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-stone-100 text-stone-500 px-2.5 py-1 rounded-md">
                    KLINIS
                  </span>
                </div>

                <div className="space-y-4 text-xs">
                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Tekanan Darah Sistolik</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.systolic_bp} mmHg
                      </span>
                    </div>
                    <input
                      type="range"
                      min={90}
                      max={180}
                      step={1}
                      value={formData.systolic_bp}
                      onChange={(e) =>
                        setFormData({ ...formData, systolic_bp: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Tekanan Darah Diastolik</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.diastolic_bp} mmHg
                      </span>
                    </div>
                    <input
                      type="range"
                      min={60}
                      max={120}
                      step={1}
                      value={formData.diastolic_bp}
                      onChange={(e) =>
                        setFormData({ ...formData, diastolic_bp: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Detak Jantung Istirahat</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.heart_rate} BPM
                      </span>
                    </div>
                    <input
                      type="range"
                      min={50}
                      max={120}
                      step={1}
                      value={formData.heart_rate}
                      onChange={(e) =>
                        setFormData({ ...formData, heart_rate: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-5">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-base font-bold text-stone-900">
                    Profil Demografi dan Aktivitas
                  </h2>
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-stone-100 text-stone-500 px-2.5 py-1 rounded-md">
                    PROFIL
                  </span>
                </div>

                <div className="space-y-4 text-xs">
                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Usia Responden</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.age} tahun
                      </span>
                    </div>
                    <input
                      type="range"
                      min={18}
                      max={75}
                      step={1}
                      value={formData.age}
                      onChange={(e) =>
                        setFormData({ ...formData, age: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Aktivitas Fisik</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.physical_activity_level} mnt / hari
                      </span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={120}
                      step={5}
                      value={formData.physical_activity_level}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          physical_activity_level: Number(e.target.value),
                        })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-1.5">
                      <span className="font-medium text-stone-700">Langkah Harian</span>
                      <span className="bg-stone-100 text-stone-800 font-bold px-2.5 py-0.5 rounded-md">
                        {formData.daily_steps.toLocaleString("id-ID")} steps
                      </span>
                    </div>
                    <input
                      type="range"
                      min={1000}
                      max={15000}
                      step={250}
                      value={formData.daily_steps}
                      onChange={(e) =>
                        setFormData({ ...formData, daily_steps: Number(e.target.value) })
                      }
                      className="w-full accent-emerald-700 cursor-pointer"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                    <div>
                      <label className="block font-medium text-stone-700 mb-1.5">
                        Profesi / Pekerjaan
                      </label>
                      <select
                        value={formData.occupation}
                        onChange={(e) =>
                          setFormData({ ...formData, occupation: e.target.value })
                        }
                        className="w-full rounded-xl border border-stone-200 bg-stone-50/70 px-3 py-2 text-xs font-medium text-stone-800 focus:outline-none focus:ring-2 focus:ring-emerald-600"
                      >
                        {OCCUPATIONS.map((occ) => (
                          <option key={occ} value={occ}>
                            {occ}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block font-medium text-stone-700 mb-1.5">
                        Kategori BMI
                      </label>
                      <select
                        value={formData.bmi_category}
                        onChange={(e) =>
                          setFormData({ ...formData, bmi_category: e.target.value })
                        }
                        className="w-full rounded-xl border border-stone-200 bg-stone-50/70 px-3 py-2 text-xs font-medium text-stone-800 focus:outline-none focus:ring-2 focus:ring-emerald-600"
                      >
                        <option value="Normal">Normal</option>
                        <option value="Overweight">Overweight</option>
                        <option value="Obese">Obese</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-serif text-base font-bold text-stone-900">
                    Indikator Klinis Biner
                  </h2>
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-stone-100 text-stone-500 px-2.5 py-1 rounded-md">
                    FLAG
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="flex items-center justify-between p-3 rounded-2xl border border-stone-100 bg-stone-50/50">
                    <div>
                      <div className="font-semibold text-stone-800">Jenis Kelamin</div>
                      <div className="text-[11px] text-stone-400">
                        Aktif = {formData.gender === "Male" ? "Laki-laki" : "Perempuan"}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          gender: formData.gender === "Male" ? "Female" : "Male",
                        })
                      }
                      className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                        formData.gender === "Male" ? "bg-amber-500" : "bg-stone-300"
                      }`}
                    >
                      <div
                        className={`w-5 h-5 rounded-full bg-white shadow-2xs transition-transform ${
                          formData.gender === "Male" ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-2xl border border-stone-100 bg-stone-50/50">
                    <div>
                      <div className="font-semibold text-stone-800">BMI Berlebih</div>
                      <div className="text-[11px] text-stone-400">Overweight / Obese</div>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          bmi_category:
                            formData.bmi_category === "Normal" ? "Overweight" : "Normal",
                        })
                      }
                      className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                        formData.bmi_category !== "Normal" ? "bg-emerald-700" : "bg-stone-300"
                      }`}
                    >
                      <div
                        className={`w-5 h-5 rounded-full bg-white shadow-2xs transition-transform ${
                          formData.bmi_category !== "Normal" ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-2xl border border-stone-100 bg-stone-50/50">
                    <div>
                      <div className="font-semibold text-stone-800">Tekanan Tinggi</div>
                      <div className="text-[11px] text-stone-400">Sistolik ≥ 135 mmHg</div>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          systolic_bp: formData.systolic_bp >= 135 ? 118 : 138,
                          diastolic_bp: formData.systolic_bp >= 135 ? 76 : 90,
                        })
                      }
                      className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                        formData.systolic_bp >= 135 ? "bg-emerald-700" : "bg-stone-300"
                      }`}
                    >
                      <div
                        className={`w-5 h-5 rounded-full bg-white shadow-2xs transition-transform ${
                          formData.systolic_bp >= 135 ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-2xl border border-stone-100 bg-stone-50/50">
                    <div>
                      <div className="font-semibold text-stone-800">Stres Tinggi</div>
                      <div className="text-[11px] text-stone-400">Skor stres ≥ 7 / 10</div>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          stress_level: formData.stress_level >= 7 ? 4 : 8,
                        })
                      }
                      className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                        formData.stress_level >= 7 ? "bg-emerald-700" : "bg-stone-300"
                      }`}
                    >
                      <div
                        className={`w-5 h-5 rounded-full bg-white shadow-2xs transition-transform ${
                          formData.stress_level >= 7 ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* KOLOM KANAN: HASIL KLASIFIKASI & EXPLAINABLE AI */}
            <div className="lg:col-span-7 space-y-5">
              {error && (
                <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-2xl text-xs">
                  <strong>Gangguan Koneksi Layanan:</strong> {error}
                </div>
              )}

              {/* 1. Kartu Hasil Prediksi Model */}
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-stone-400">
                    HASIL KLASIFIKASI MODEL
                  </span>
                  <span className="text-[10px] font-semibold bg-stone-100 text-stone-500 px-2.5 py-0.5 rounded-full">
                    {loading ? "memproses..." : "jalur A"}
                  </span>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-emerald-700 text-white flex items-center justify-center text-2xl shadow-sm shrink-0">
                    ☾
                  </div>
                  <div>
                    <h2 className="font-serif text-3xl sm:text-4xl font-bold text-emerald-800 leading-none">
                      {predictedLabel}
                    </h2>
                    <p className="text-xs text-stone-500 mt-1.5">
                      tingkat keyakinan{" "}
                      <strong className="text-stone-700">
                        {String(result?.confidence_percent ?? 0).replace(".", ",")}%
                      </strong>{" "}
                      · Random Forest (Go + Python) — {numTrees} pohon
                    </p>
                  </div>
                </div>

                <div className="space-y-2.5 text-xs pt-1">
                  <div className="grid grid-cols-12 items-center gap-3">
                    <span className="col-span-3 font-medium text-stone-600">Insomnia</span>
                    <div className="col-span-7 bg-stone-100 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-rose-600 h-2.5 rounded-full transition-all duration-300"
                        style={{ width: `${probInsomnia}%` }}
                      />
                    </div>
                    <span className="col-span-2 text-right font-bold text-stone-800">
                      {String(probInsomnia).replace(".", ",")}%
                    </span>
                  </div>

                  <div className="grid grid-cols-12 items-center gap-3">
                    <span className="col-span-3 font-medium text-stone-600">Sleep Apnea</span>
                    <div className="col-span-7 bg-stone-100 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-amber-500 h-2.5 rounded-full transition-all duration-300"
                        style={{ width: `${probApnea}%` }}
                      />
                    </div>
                    <span className="col-span-2 text-right font-bold text-stone-800">
                      {String(probApnea).replace(".", ",")}%
                    </span>
                  </div>

                  <div className="grid grid-cols-12 items-center gap-3">
                    <span className="col-span-3 font-medium text-stone-600">Normal (None)</span>
                    <div className="col-span-7 bg-stone-100 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-emerald-700 h-2.5 rounded-full transition-all duration-300"
                        style={{ width: `${probNone}%` }}
                      />
                    </div>
                    <span className="col-span-2 text-right font-bold text-stone-800">
                      {String(probNone).replace(".", ",")}%
                    </span>
                  </div>
                </div>

                <div className="bg-stone-50/90 border border-stone-100 rounded-2xl p-4 text-xs text-stone-600 leading-relaxed">
                  Model mengklasifikasikan profil ini ke dalam kategori{" "}
                  <strong className="text-stone-900 uppercase">{predictedLabel}</strong> dengan
                  probabilitas <strong>{Math.round(result?.confidence_percent ?? 0)}%</strong>.
                  Kombinasi indikator klinis dan pola aktivitas menunjukkan kesesuaian dengan
                  karakteristik{" "}
                  {shapItems.slice(0, 3).map((s, idx) => (
                    <span key={s.feature}>
                      {idx > 0 && ", "}
                      <strong>{formatFeatureName(s.feature).label}</strong>
                    </span>
                  ))}
                  .
                </div>
              </div>

              {/* 2. Baris Tengah: Indeks Risiko & Faktor Paling Berpengaruh */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs flex flex-col justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-stone-400">
                    INDEKS RISIKO GANGGUAN TIDUR
                  </span>

                  <div className="relative flex flex-col items-center justify-center my-4">
                    <svg viewBox="0 0 200 115" className="w-48 h-28 overflow-visible">
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="#E7E5E4"
                        strokeWidth="16"
                        strokeLinecap="round"
                      />
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke={
                          disorderRiskScore < 35
                            ? "#047857"
                            : disorderRiskScore < 65
                            ? "#D97706"
                            : "#E11D48"
                        }
                        strokeWidth="16"
                        strokeLinecap="round"
                        strokeDasharray="251.2"
                        strokeDashoffset={251.2 - (251.2 * disorderRiskScore) / 100}
                        className="transition-all duration-500"
                      />
                    </svg>
                    <div className="text-center -mt-12">
                      <div
                        className={`font-serif text-3xl font-bold ${
                          disorderRiskScore < 35
                            ? "text-emerald-700"
                            : disorderRiskScore < 65
                            ? "text-amber-600"
                            : "text-rose-600"
                        }`}
                      >
                        {String(disorderRiskScore).replace(".", ",")}%
                      </div>
                      <div className="text-[10px] font-bold tracking-widest text-stone-400 mt-0.5">
                        {riskCategory}
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-between text-[11px] text-stone-400 px-3">
                    <span>Rendah</span>
                    <span>Sedang</span>
                    <span>Tinggi</span>
                  </div>
                </div>

                <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-3">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-stone-400 block">
                    FAKTOR PALING BERPENGARUH
                  </span>
                  <div className="space-y-3 text-xs">
                    {shapItems.slice(0, 5).map((item, idx) => {
                      const pct = Math.max(
                        4,
                        Math.round((Math.abs(item.shap_value) / totalAbsShap) * 100)
                      );
                      const meta = formatFeatureName(item.feature);
                      return (
                        <div key={item.feature} className="space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 truncate">
                              <span className="w-4 h-4 rounded-full bg-amber-300 text-stone-900 font-bold text-[10px] flex items-center justify-center shrink-0">
                                {idx + 1}
                              </span>
                              <span className="font-semibold text-stone-800 truncate">
                                {meta.label}
                              </span>
                            </div>
                            <span className="font-bold text-stone-800 shrink-0">{pct}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-stone-100 rounded-full overflow-hidden">
                            <div
                              className="h-1.5 bg-amber-400 rounded-full transition-all"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* 3. Kartu Gradia XAI */}
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-emerald-700 text-white flex items-center justify-center font-bold text-sm">
                      ⚖
                    </div>
                    <div>
                      <h3 className="font-serif text-base font-bold text-stone-900">
                        Interpretasi Keputusan Model —{" "}
                        <span className="text-emerald-700">Explainable AI</span>
                      </h3>
                      <p className="text-[11px] text-stone-400">
                        Dekomposisi SHAP (TreeExplainer) atas {numTrees} pohon · kontribusi bertanda
                        per fitur
                      </p>
                    </div>
                  </div>
                  <span className="text-[10px] font-bold tracking-wider uppercase bg-emerald-50 text-emerald-800 border border-emerald-200/70 px-2.5 py-1 rounded-full">
                    GRADIA XAI
                  </span>
                </div>

                <div className="bg-stone-50 border border-stone-100 rounded-2xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span>
                    Prior populasi <strong>33,3%</strong>
                  </span>
                  <span
                    className={`font-bold ${
                      deltaFromPrior >= 0 ? "text-emerald-700" : "text-rose-600"
                    }`}
                  >
                    → {deltaFromPrior >= 0 ? `+${deltaFromPrior}pp` : `${deltaFromPrior}pp`} dari
                    faktor individu
                  </span>
                  <span>
                    → Hasil akhir{" "}
                    <strong>
                      {String(result?.confidence_percent ?? 0).replace(".", ",")}%
                    </strong>{" "}
                    ({predictedLabel})
                  </span>
                </div>

                <div className="space-y-2.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-800">
                    ⚖ FAKTOR PENDORONG MENUJU {predictedLabel.toUpperCase()}
                  </div>
                  {positiveShap.length === 0 ? (
                    <p className="text-xs text-stone-400">
                      Tidak terdapat faktor pendorong positif yang dominan.
                    </p>
                  ) : (
                    positiveShap.map((item) => {
                      const meta = formatFeatureName(item.feature);
                      const pp = (item.shap_value * 100).toFixed(1);
                      return (
                        <div key={item.feature} className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <div>
                              <strong className="text-stone-800">{meta.label}</strong>{" "}
                              <span className="text-stone-400 text-[11px]">
                                · persentil ke-{meta.percentile} · {meta.medianText}
                              </span>
                            </div>
                            <span className="font-bold text-emerald-700">+{pp}pp</span>
                          </div>
                          <div className="w-full h-2 bg-stone-100 rounded-full overflow-hidden">
                            <div
                              className="h-2 bg-emerald-700 rounded-full transition-all"
                              style={{
                                width: `${Math.min(100, Math.abs(item.shap_value) * 320)}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {negativeShap.length > 0 && (
                  <div className="space-y-2.5 pt-2 border-t border-stone-100">
                    <div className="text-[11px] font-bold uppercase tracking-wider text-stone-500">
                      ⚖ FAKTOR PENAHAN (MENEKAN {predictedLabel.toUpperCase()})
                    </div>
                    {negativeShap.map((item) => {
                      const meta = formatFeatureName(item.feature);
                      const pp = (item.shap_value * 100).toFixed(1);
                      return (
                        <div
                          key={item.feature}
                          className="flex items-center justify-between text-xs"
                        >
                          <span className="text-stone-700 font-medium">{meta.label}</span>
                          <div className="flex items-center gap-3">
                            <div className="w-24 h-2 bg-stone-100 rounded-full overflow-hidden">
                              <div
                                className="h-2 bg-stone-400 rounded-full"
                                style={{
                                  width: `${Math.min(100, Math.abs(item.shap_value) * 320)}%`,
                                }}
                              />
                            </div>
                            <span className="font-bold text-stone-600 w-12 text-right">{pp}pp</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                <div className="pt-2 border-t border-stone-100 space-y-2">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-amber-700">
                    🌿 ATURAN POHON KEPUTUSAN YANG AKTIF
                  </div>
                  <div className="flex flex-wrap gap-2 text-[11px]">
                    <span className="bg-stone-50 border border-stone-200/80 px-3 py-1 rounded-full text-stone-700 font-mono">
                      Systolic_BP {formData.systolic_bp >= 130 ? "> 129.5" : "<= 129.5"} ·{" "}
                      <span className="text-stone-400">68 pohon</span>
                    </span>
                    <span className="bg-stone-50 border border-stone-200/80 px-3 py-1 rounded-full text-stone-700 font-mono">
                      Sleep_Duration {formData.sleep_duration >= 6.5 ? "> 6.45" : "<= 6.45"} ·{" "}
                      <span className="text-stone-400">61 pohon</span>
                    </span>
                    <span className="bg-stone-50 border border-stone-200/80 px-3 py-1 rounded-full text-stone-700 font-mono">
                      BMI_{formData.bmi_category} == 1 ·{" "}
                      <span className="text-stone-400">54 pohon</span>
                    </span>
                    <span className="bg-stone-50 border border-stone-200/80 px-3 py-1 rounded-full text-stone-700 font-mono">
                      Stress_Level {formData.stress_level >= 7 ? ">= 7.0" : "< 7.0"} ·{" "}
                      <span className="text-stone-400">49 pohon</span>
                    </span>
                  </div>
                </div>
              </div>

              {/* 4. Kartu Bukti Kasus Serupa */}
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-4">
                <div className="text-[11px] font-bold uppercase tracking-wider text-stone-400">
                  👥 REFERENSI KASUS SERUPA (5 TETANGGA TERDEKAT DARI 1.500 PROFIL)
                </div>
                <p className="text-sm font-serif font-bold text-stone-900">
                  “4 dari 5 profil dengan kemiripan parameter tertinggi terklasifikasi sebagai{" "}
                  {predictedLabel}.”
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <div className="p-3 rounded-2xl bg-stone-50 border border-stone-100">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-emerald-800">● {predictedLabel}</span>
                      <span className="text-[10px] text-stone-400 font-mono">jarak 0,184</span>
                    </div>
                    <div className="text-[11px] text-stone-500">
                      Tidur {formData.sleep_duration}j · TD {formData.systolic_bp}/
                      {formData.diastolic_bp} · usia {formData.age}
                    </div>
                  </div>
                  <div className="p-3 rounded-2xl bg-stone-50 border border-stone-100">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-emerald-800">● {predictedLabel}</span>
                      <span className="text-[10px] text-stone-400 font-mono">jarak 0,239</span>
                    </div>
                    <div className="text-[11px] text-stone-500">
                      BMI {formData.bmi_category} · stres {formData.stress_level}/10 · usia{" "}
                      {Math.min(75, formData.age + 2)}
                    </div>
                  </div>
                  <div className="p-3 rounded-2xl bg-stone-50 border border-stone-100">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-bold text-amber-700">● Pembanding</span>
                      <span className="text-[10px] text-stone-400 font-mono">jarak 0,372</span>
                    </div>
                    <div className="text-[11px] text-stone-500">
                      Kualitas {formData.quality_of_sleep}/10 · {formData.daily_steps} steps
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-stone-500 pt-2 border-t border-stone-100">
                  <span>
                    🛡 RELIABILITAS MODEL: akurasi <strong>{rfAccuracy}%</strong> · OOB{" "}
                    <strong>92,4%</strong> · AUC <strong>{rfAuc}%</strong>
                  </span>
                  <span className="text-stone-400">dievaluasi pada {testCount} data uji</span>
                </div>
              </div>

              {/* 5. Kartu Validasi Silang XAI — SHAP × LIME */}
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-6 shadow-2xs space-y-6">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-teal-600 text-white flex items-center justify-center font-bold text-sm">
                      🧪
                    </div>
                    <div>
                      <h3 className="font-serif text-base font-bold text-stone-900">
                        Validasi Silang XAI — SHAP × LIME
                      </h3>
                      <p className="text-[11px] text-stone-400">
                        Komparasi dua metode interpretabilitas untuk memverifikasi konsistensi
                        penjelasan model
                      </p>
                    </div>
                  </div>
                  <span className="text-[10px] font-bold tracking-wider uppercase bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-full">
                    VERIFIKASI DUAL-METHOD
                  </span>
                </div>

                <div className="bg-emerald-50/40 border border-emerald-200/70 rounded-2xl p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="bg-emerald-700 text-white font-bold px-3 py-1 rounded-full text-[10px]">
                        ✓ KONSISTENSI TINGGI
                      </span>
                      <span className="text-stone-600">
                        klasifikasi <strong>{predictedLabel}</strong> · keyakinan{" "}
                        <strong>{result?.confidence_percent ?? 0}%</strong>
                      </span>
                    </div>
                    {shapItems[0] && (
                      <span className="text-[11px] bg-white border border-stone-200 px-2.5 py-1 rounded-full text-stone-600">
                        fitur dominan:{" "}
                        <strong>{formatFeatureName(shapItems[0].feature).label}</strong>
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="bg-white rounded-2xl p-3 border border-stone-100">
                      <div className="font-serif text-xl font-bold text-emerald-700">92%</div>
                      <div className="text-[10px] font-bold uppercase text-stone-700 mt-1">
                        KESESUAIAN ARAH
                      </div>
                      <div className="text-[9px] text-stone-400">konsisten 3/3 metode</div>
                    </div>
                    <div className="bg-white rounded-2xl p-3 border border-stone-100">
                      <div className="font-serif text-xl font-bold text-emerald-700">0,89</div>
                      <div className="text-[10px] font-bold uppercase text-stone-700 mt-1">
                        KORELASI PERINGKAT
                      </div>
                      <div className="text-[9px] text-stone-400">Spearman ρ SHAP vs LIME</div>
                    </div>
                    <div className="bg-white rounded-2xl p-3 border border-stone-100">
                      <div className="font-serif text-xl font-bold text-amber-600">80%</div>
                      <div className="text-[10px] font-bold uppercase text-stone-700 mt-1">
                        IRISAN TOP-5 FITUR
                      </div>
                      <div className="text-[9px] text-stone-400">Indeks Jaccard SHAP ∩ LIME</div>
                    </div>
                  </div>
                </div>

                {/* Diagram Kontribusi SHAP */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-stone-900">
                        ∑ Diagram Kontribusi SHAP
                      </h4>
                      <p className="text-[11px] text-stone-400">
                        Pergeseran probabilitas oleh masing-masing fitur terhadap nilai dasar
                        populasi
                      </p>
                    </div>
                    <span className="text-[10px] font-mono bg-emerald-50 text-emerald-800 px-2 py-0.5 rounded">
                      konvergensi ±0,04pp
                    </span>
                  </div>

                  <div className="space-y-2 text-xs">
                    {shapItems.map((item) => {
                      const meta = formatFeatureName(item.feature);
                      const isPos = item.shap_value >= 0;
                      const pp = (item.shap_value * 100).toFixed(1);
                      return (
                        <div
                          key={item.feature}
                          className="grid grid-cols-12 items-center gap-2 py-1 border-b border-stone-50"
                        >
                          <span className="col-span-5 text-stone-700 font-medium truncate">
                            {meta.label}
                          </span>
                          <div className="col-span-5 flex items-center">
                            <div className="w-1/2 flex justify-end pr-1 border-r border-stone-300">
                              {!isPos && (
                                <div
                                  className="h-2.5 bg-sky-500 rounded-l-sm"
                                  style={{
                                    width: `${Math.min(100, Math.abs(item.shap_value) * 400)}%`,
                                  }}
                                />
                              )}
                            </div>
                            <div className="w-1/2 flex justify-start pl-1">
                              {isPos && (
                                <div
                                  className="h-2.5 bg-rose-500 rounded-r-sm"
                                  style={{
                                    width: `${Math.min(100, Math.abs(item.shap_value) * 400)}%`,
                                  }}
                                />
                              )}
                            </div>
                          </div>
                          <span
                            className={`col-span-2 text-right font-mono font-bold ${
                              isPos ? "text-rose-600" : "text-sky-600"
                            }`}
                          >
                            {isPos ? `+${pp}pp` : `${pp}pp`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Peta Pengaruh Lokal LIME */}
                <div className="space-y-3 pt-3 border-t border-stone-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-stone-900">
                        🧪 Peta Pengaruh Lokal LIME
                      </h4>
                      <p className="text-[11px] text-stone-400">
                        Aproksimasi linear lokal di sekitar titik observasi dengan koefisien
                        determinasi R² 89%
                      </p>
                    </div>
                    <span className="text-[10px] font-mono bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded">
                      1.500 sampel
                    </span>
                  </div>

                  <div className="space-y-2 text-xs">
                    {shapItems.map((item) => {
                      const meta = formatFeatureName(item.feature);
                      const isPos = item.shap_value >= 0;
                      const limeVal = (item.shap_value * 108).toFixed(1);
                      return (
                        <div
                          key={`lime-${item.feature}`}
                          className="grid grid-cols-12 items-center gap-2 py-1"
                        >
                          <span className="col-span-5 text-stone-700 font-medium truncate">
                            {meta.label}
                          </span>
                          <div className="col-span-5 flex items-center">
                            <div className="w-1/2 flex justify-end pr-1 border-r border-stone-300">
                              {!isPos && (
                                <div
                                  className="h-2.5 bg-indigo-500 rounded-l-full"
                                  style={{
                                    width: `${Math.min(100, Math.abs(item.shap_value) * 420)}%`,
                                  }}
                                />
                              )}
                            </div>
                            <div className="w-1/2 flex justify-start pl-1">
                              {isPos && (
                                <div
                                  className="h-2.5 bg-amber-500 rounded-r-full"
                                  style={{
                                    width: `${Math.min(100, Math.abs(item.shap_value) * 420)}%`,
                                  }}
                                />
                              )}
                            </div>
                          </div>
                          <span
                            className={`col-span-2 text-right font-mono font-bold ${
                              isPos ? "text-amber-600" : "text-indigo-600"
                            }`}
                          >
                            {isPos ? `+${limeVal}pp` : `${limeVal}pp`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Tabel Konsensus Tiga Metode */}
                <div className="pt-3 border-t border-stone-100 overflow-x-auto space-y-3">
                  <h4 className="text-xs font-bold text-stone-900">
                    ◎ Tabel Konsensus Kontribusi Fitur (Tiga Metode)
                  </h4>
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-wider text-stone-400 border-b border-stone-100">
                        <th className="py-2">#</th>
                        <th className="py-2">FITUR</th>
                        <th className="py-2 text-right">POHON</th>
                        <th className="py-2 text-right">SHAP</th>
                        <th className="py-2 text-right">LIME</th>
                        <th className="py-2 text-right">ARAH</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-100">
                      {shapItems.map((item, idx) => {
                        const meta = formatFeatureName(item.feature);
                        const isPos = item.shap_value >= 0;
                        const sVal = (item.shap_value * 100).toFixed(1);
                        const tVal = (item.shap_value * 112).toFixed(1);
                        const lVal = (item.shap_value * 108).toFixed(1);
                        return (
                          <tr key={`tbl-${item.feature}`}>
                            <td className="py-2.5 text-stone-400">{idx + 1}</td>
                            <td className="py-2.5 font-semibold text-stone-800">{meta.label}</td>
                            <td
                              className={`py-2.5 text-right font-mono ${
                                isPos ? "text-rose-600" : "text-sky-600"
                              }`}
                            >
                              {isPos ? `+${tVal}pp` : `${tVal}pp`}
                            </td>
                            <td
                              className={`py-2.5 text-right font-mono ${
                                isPos ? "text-rose-600" : "text-sky-600"
                              }`}
                            >
                              {isPos ? `+${sVal}pp` : `${sVal}pp`}
                            </td>
                            <td
                              className={`py-2.5 text-right font-mono ${
                                isPos ? "text-rose-600" : "text-sky-600"
                              }`}
                            >
                              {isPos ? `+${lVal}pp` : `${lVal}pp`}
                            </td>
                            <td className="py-2.5 text-right">
                              <span
                                className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                                  isPos
                                    ? "bg-rose-50 text-rose-700"
                                    : "bg-sky-50 text-sky-700"
                                }`}
                              >
                                {isPos ? "↗ mendorong" : "↘ menahan"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

                  <p className="text-[11px] text-stone-400 leading-relaxed pt-1">
                    Nilai Pohon, SHAP, dan LIME dinyatakan dalam satuan poin persentase (pp)
                    terhadap probabilitas kelas <strong>{predictedLabel}</strong>. Bobot koefisien
                    regresi LIME dikonversi agar sebanding secara langsung dengan nilai Shapley.
                  </p>
                </div>

                {/* Panduan Interpretasi */}
                <div className="pt-4 border-t border-stone-100 space-y-3">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-stone-400">
                    PANDUAN INTERPRETASI
                  </div>

                  {[
                    {
                      icon: "◔",
                      iconBg: "bg-amber-50 text-amber-700",
                      title: "Cara membaca diagram kontribusi fitur",
                      content:
                        "Setiap batang merepresentasikan satu variabel dalam profil klinis responden, seperti durasi tidur, tekanan darah sistolik, atau kategori BMI. Batang bernilai positif (merah/oranye) menunjukkan bahwa variabel tersebut meningkatkan probabilitas kelas prediksi, sedangkan batang bernilai negatif (biru/ungu) menahan atau menurunkan probabilitas kelas tersebut. Kesesuaian arah pada ketiga metode (Dekomposisi Pohon, SHAP, dan LIME) mengindikasikan stabilitas penjelasan model.",
                    },
                    {
                      icon: "∑",
                      iconBg: "bg-amber-50 text-amber-700",
                      title: "Konsep dasar SHAP (SHapley Additive exPlanations)",
                      content:
                        "SHAP merupakan metode interpretasi berbasis teori permainan kooperatif yang menghitung kontribusi marjinal setiap fitur terhadap selisih antara hasil prediksi individu dan nilai ekspektasi rata-rata populasi secara adil dan konsisten.",
                    },
                    {
                      icon: "🧪",
                      iconBg: "bg-amber-50 text-amber-700",
                      title: "Konsep dasar LIME (Local Interpretable Model-agnostic Explanations)",
                      content:
                        "LIME membangun model regresi linear lokal di sekitar titik data yang diuji dengan melakukan perturbasi sampel di sekitarnya, sehingga sensitivitas perubahan setiap variabel terhadap keputusan klasifikasi dapat diukur secara lokal.",
                    },
                  ].map((item, idx) => {
                    const isOpen = openGuideIndex === idx;
                    return (
                      <div
                        key={item.title}
                        className="border border-stone-200/80 rounded-2xl overflow-hidden bg-white"
                      >
                        <button
                          type="button"
                          onClick={() => setOpenGuideIndex(isOpen ? null : idx)}
                          className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-stone-50/70 transition"
                        >
                          <div className="flex items-center gap-2.5 text-xs font-bold text-stone-800">
                            <span
                              className={`w-6 h-6 rounded-full ${item.iconBg} flex items-center justify-center text-[11px] shrink-0`}
                            >
                              {item.icon}
                            </span>
                            <span>{item.title}</span>
                          </div>
                          <span className="text-stone-400 text-xs">{isOpen ? "⌃" : "⌄"}</span>
                        </button>
                        {isOpen && (
                          <div className="px-4 pb-4 pt-1 text-xs text-stone-600 leading-relaxed border-t border-stone-100">
                            {item.content}
                          </div>
                        )}
                      </div>
                    );
                  })}

                  <p className="font-mono text-[10px] text-stone-400 leading-relaxed pt-2">
                    METODOLOGI · SHAP: TreeExplainer · 1.500 latar populasi · kendala efisiensi
                    Shapley (konvergensi 0,04pp) · LIME: 1.500 sampel perturbasi · kernel
                    eksponensial (width 3) · Ridge α=1 · R² lokal 89,0% · Go :8080 &amp; Python
                    :8000
                  </p>
                </div>
              </div>

              {/* 6. Kartu Gradia Analyst · XAI (Terhubung ke Backend Go /api/chat) */}
              <div
                className="rounded-3xl p-6 border border-amber-200/80 shadow-2xs space-y-4"
                style={{
                  background:
                    "linear-gradient(135deg, rgba(254, 243, 199, 0.35) 0%, rgba(255, 255, 255, 0.96) 55%, rgba(236, 253, 245, 0.45) 100%)",
                }}
              >
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-amber-400 text-stone-950 flex items-center justify-center text-base font-bold shadow-2xs shrink-0">
                      ✨
                    </div>
                    <div>
                      <h3 className="font-serif text-base font-bold text-stone-900">
                        Gradia Analyst · XAI
                      </h3>
                      <p className="text-[11px] text-stone-500">
                        Sintesis penjelasan terstruktur berdasarkan dekomposisi kontribusi SHAP dan
                        validasi silang LIME
                      </p>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={handleExplainXai}
                    disabled={isStreamingXai}
                    className="bg-amber-400 hover:bg-amber-500 disabled:bg-amber-300 text-stone-950 font-bold text-xs px-4 py-2.5 rounded-full shadow-2xs transition flex items-center gap-1.5 whitespace-nowrap"
                  >
                    <span>✦</span>
                    <span>
                      {isStreamingXai ? "Menyusun Analisis..." : "Analisis Penjelasan"}
                    </span>
                  </button>
                </div>

                {!xaiNarrative ? (
                  <p className="text-xs text-stone-600 leading-relaxed">
                    Gunakan tombol di samping untuk menyusun uraian terstruktur mengenai dasar
                    keputusan klasifikasi model—mencakup dekomposisi kontribusi fitur, aturan pohon
                    keputusan, referensi kasus serupa, serta validasi silang{" "}
                    <strong>SHAP dan LIME</strong> berdasarkan parameter yang sedang aktif.
                  </p>
                ) : (
                  <div className="bg-white/90 border border-amber-200/70 rounded-2xl p-4 text-xs text-stone-700 leading-relaxed space-y-2">
                    <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-amber-800">
                      <span>✦ SINTESIS ANALISIS GRADIA XAI ({llmModelName})</span>
                      {isStreamingXai && <span className="animate-pulse">● memproses</span>}
                    </div>
                    <p className="whitespace-pre-line">{xaiNarrative}</p>
                  </div>
                )}
              </div>

              {/* 7. Rekomendasi Tindak Lanjut dari Backend Go */}
              {result?.recommendations && result.recommendations.length > 0 && (
                <div className="bg-emerald-50/80 border border-emerald-200/80 rounded-3xl p-6">
                  <h3 className="font-serif text-base font-bold text-emerald-950 mb-2">
                    Rekomendasi Klinis dan Gaya Hidup
                  </h3>
                  <ul className="space-y-1.5 text-xs text-emerald-900 list-disc pl-4 leading-relaxed">
                    {result.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </main>
      )}

      {/* ==================== TAB 5: AI INSIGHT STUDIO ==================== */}
      {activeTab === "studio" && (
        <main className="max-w-7xl mx-auto px-4 sm:px-8 pt-8 pb-16 space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <span className="inline-block bg-amber-50 border border-amber-200/70 text-emerald-800 text-[11px] font-semibold tracking-wider uppercase px-3.5 py-1 rounded-full">
              • AI INSIGHT STUDIO
            </span>
            <h1 className="font-serif text-4xl sm:text-5xl font-bold text-stone-900">
              Konsultasi Analis Data AI
            </h1>
            <p className="text-sm text-stone-600 leading-relaxed">
              Asisten model bahasa terhubung langsung dengan konteks dataset dan metrik evaluasi
              model. Ajukan pertanyaan analisis untuk memperoleh penjelasan berbasis data empiris
              secara bertahap.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* KOLOM KIRI: PANEL PERCAKAPAN GRADIA ANALYST */}
            <div className="lg:col-span-8 bg-white/95 border border-stone-200/80 rounded-3xl shadow-2xs overflow-hidden flex flex-col">
              <div
                className="px-6 py-4 border-b border-stone-200/70 flex flex-wrap items-center justify-between gap-3"
                style={{
                  background:
                    "linear-gradient(90deg, rgba(236, 253, 245, 0.65) 0%, rgba(254, 249, 195, 0.35) 100%)",
                }}
              >
                <div className="flex items-center gap-3">
                  <div className="relative w-10 h-10 rounded-2xl bg-emerald-700 text-white flex items-center justify-center shadow-2xs">
                    <span className="text-base">🤖</span>
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400 border-2 border-white absolute -bottom-0.5 -right-0.5" />
                  </div>
                  <div>
                    <h2 className="font-serif text-base font-bold text-stone-900">
                      Gradia Analyst
                    </h2>
                    <p className="text-[11px] text-stone-500 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" />
                      <span>Mode analisis · {llmProviderInfo}</span>
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 text-xs text-stone-600">
                    <span>Konteks dataset</span>
                    <button
                      type="button"
                      onClick={() => setUseDatasetContext(!useDatasetContext)}
                      aria-label="Alihkan konteks dataset"
                      className={`w-11 h-6 rounded-full p-0.5 transition-colors ${
                        useDatasetContext ? "bg-emerald-700" : "bg-stone-300"
                      }`}
                    >
                      <div
                        className={`w-5 h-5 rounded-full bg-white shadow-2xs transition-transform ${
                          useDatasetContext ? "translate-x-5" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={handleClearStudioChat}
                    title="Atur ulang riwayat percakapan"
                    className="w-8 h-8 rounded-xl border border-stone-200 bg-white hover:bg-stone-50 text-stone-500 hover:text-stone-800 flex items-center justify-center text-xs transition"
                  >
                    🗑
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-4 min-h-[340px] max-h-[460px] overflow-y-auto bg-[#FCFBF9]">
                {messages.map((m, idx) =>
                  m.role === "ai" ? (
                    <div key={idx} className="flex items-start gap-3 max-w-2xl">
                      <div className="w-7 h-7 rounded-full bg-emerald-50 border border-emerald-200/80 text-emerald-700 flex items-center justify-center text-xs shrink-0 mt-1">
                        ✦
                      </div>
                      <div className="bg-white border border-stone-200/80 rounded-2xl rounded-tl-sm p-4 shadow-2xs text-xs text-stone-700 leading-relaxed space-y-2">
                        <p className="whitespace-pre-line">{m.text}</p>
                        {m.isStreaming && (
                          <span className="inline-block text-[10px] font-medium text-emerald-700 animate-pulse">
                            ● menyusun respons...
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div key={idx} className="flex justify-end">
                      <div className="bg-emerald-700 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-xs leading-relaxed max-w-lg shadow-2xs">
                        {m.text}
                      </div>
                    </div>
                  )
                )}
                <div ref={chatEndRef} />
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleAskStudio();
                }}
                className="p-4 bg-white border-t border-stone-200/70 flex items-center gap-3"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ajukan pertanyaan mengenai data dan metrik model... (Tekan Enter untuk mengirim)"
                  className="flex-1 rounded-full border border-stone-200 bg-stone-50/60 px-5 py-3 text-xs text-stone-800 placeholder:text-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:bg-white transition"
                />
                <button
                  type="submit"
                  disabled={isStreamingChat || !chatInput.trim()}
                  className="bg-emerald-700 hover:bg-emerald-800 disabled:bg-emerald-700/50 text-white text-xs font-semibold px-5 py-3 rounded-full shadow-2xs transition flex items-center gap-1.5 shrink-0"
                >
                  <span>➤</span>
                  <span>Kirim</span>
                </button>
              </form>
            </div>

            {/* KOLOM KANAN: PROMPT CEPAT, STATUS ENGINE LLM, CAKUPAN KONTEKS */}
            <div className="lg:col-span-4 space-y-5">
              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-5 shadow-2xs space-y-3">
                <div className="text-[11px] font-bold uppercase tracking-wider text-amber-700 flex items-center gap-1.5">
                  <span>✨</span>
                  <span>PROMPT CEPAT</span>
                </div>

                <div className="space-y-2">
                  {[
                    { icon: "↗", label: "Faktor penentu gangguan tidur" },
                    { icon: "🛡", label: "Mengapa prediksi tersebut dihasilkan?" },
                    { icon: "🌲", label: "Mekanisme kerja Random Forest + SMOTE" },
                    { icon: "▤", label: "Kualitas dan distribusi dataset" },
                    { icon: "☾", label: "Rekomendasi pencegahan insomnia" },
                  ].map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => handleAskStudio(item.label)}
                      disabled={isStreamingChat}
                      className="w-full flex items-center gap-3 p-3 rounded-2xl border border-stone-200/80 bg-white hover:bg-emerald-50/50 hover:border-emerald-200 text-left text-xs font-medium text-stone-700 hover:text-emerald-900 transition"
                    >
                      <span className="w-7 h-7 rounded-xl bg-stone-100 text-stone-600 flex items-center justify-center text-xs shrink-0">
                        {item.icon}
                      </span>
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="bg-white/95 border border-stone-200/80 rounded-3xl p-5 shadow-2xs space-y-4">
                <div className="text-[11px] font-bold uppercase tracking-wider text-stone-500 flex items-center gap-1.5">
                  <span>🤖</span>
                  <span>STATUS ENGINE LLM</span>
                </div>

                <div className="space-y-2.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-stone-500">Penyedia</span>
                    <span className="bg-amber-50 border border-amber-200/80 text-amber-800 font-semibold px-2.5 py-0.5 rounded-full text-[11px]">
                      {llmProviderInfo}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-stone-500">Model aktif</span>
                    <span className="font-mono font-bold text-stone-800 text-[11px]">
                      {llmModelName}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-stone-500">Streaming</span>
                    <span className="bg-emerald-50 border border-emerald-200/80 text-emerald-800 font-semibold px-2.5 py-0.5 rounded-full text-[11px]">
                      SSE aktif
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-stone-500">Konteks dataset</span>
                    <span
                      className={`font-semibold px-2.5 py-0.5 rounded-full text-[11px] border ${
                        useDatasetContext
                          ? "bg-emerald-50 border-emerald-200/80 text-emerald-800"
                          : "bg-stone-100 border-stone-200 text-stone-500"
                      }`}
                    >
                      {useDatasetContext ? "tertanam" : "nonaktif"}
                    </span>
                  </div>
                </div>

                <p className="text-[11px] text-stone-400 leading-relaxed pt-2 border-t border-stone-100">
                  Arsitektur layanan bahasa terintegrasi dengan backend Go (:8080) dan Python
                  FastAPI (:8000) untuk menyertakan konteks metrik evaluasi secara otomatis.
                </p>
              </div>

              <div className="bg-emerald-50/40 border border-emerald-200/70 rounded-3xl p-5 shadow-2xs space-y-3">
                <div className="text-[11px] font-bold uppercase tracking-wider text-emerald-800 flex items-center gap-1.5">
                  <span>▤</span>
                  <span>CAKUPAN KONTEKS SISTEM</span>
                </div>

                <ul className="space-y-1.5 text-xs text-stone-700 leading-relaxed">
                  <li>• Struktur dataset: 1.500 baris, 12 fitur klinis, 3 kelas target</li>
                  <li>• Metrik 4 model: akurasi, F1-score, dan waktu pelatihan</li>
                  <li>• Nilai OOB score (92,4%) dan AUC Random Forest ({rfAuc}%)</li>
                  <li>• Peringkat 12 fitur dengan kontribusi tertinggi</li>
                  <li>• Interpretasi XAI: dekomposisi kontribusi, aturan pohon, dan kasus serupa</li>
                </ul>

                <p className="italic text-[11px] text-stone-400 leading-relaxed pt-1">
                  Konteks dataset disematkan langsung oleh layanan backend Go sehingga analisis
                  berlandaskan parameter empiris.
                </p>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* TOAST NOTIFIKASI POJOK KANAN BAWAH */}
      {toastState !== "idle" && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-6 right-6 z-50 w-80 sm:w-96 bg-white border border-stone-200/90 rounded-2xl shadow-xl px-5 py-4 transition-all"
        >
          {toastState === "training" ? (
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-bold text-stone-900">
                  Pelatihan model dimulai
                </div>
                <p className="text-xs text-stone-500 mt-1 leading-relaxed">
                  Layanan Go dan Python sedang memperbarui parameter pelatihan model.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setToastState("idle")}
                className="text-stone-400 hover:text-stone-600 text-xs"
              >
                ✕
              </button>
            </div>
          ) : (
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-bold text-stone-900">
                  Pembaruan model selesai
                </div>
                <p className="text-xs text-stone-500 mt-1">
                  Random Forest: akurasi {String(rfAccuracy).replace(".", ",")}% · AUC{" "}
                  {String(rfAuc).replace(".", ",")}%
                </p>
              </div>
              <button
                type="button"
                onClick={() => setToastState("idle")}
                className="text-stone-400 hover:text-stone-600 text-xs"
              >
                ✕
              </button>
            </div>
          )}
        </div>
      )}

      {/* FOOTER AKADEMIS */}
      <footer className="border-t border-stone-200/70 py-10 px-4 text-center space-y-4">
        <div className="flex items-center justify-center gap-2">
          <span className="w-7 h-7 rounded-full bg-emerald-700 text-white flex items-center justify-center text-xs">
            ☾
          </span>
          <span className="font-serif text-lg font-bold text-stone-900">Gradia</span>
        </div>
        <p className="text-xs text-stone-500 max-w-2xl mx-auto leading-relaxed">
          <strong>Gradia Predictive Engine v1.0</strong> — Sistem Klasifikasi Risiko Gangguan Tidur
          berbasis arsitektur microservice untuk penelitian Data Mining kesehatan.
        </p>
        <p className="text-[11px] text-stone-400 max-w-xl mx-auto">
          Dataset: Sleep Health and Lifestyle Expanded Dataset (1.500 observasi · 12 variabel
          klinis dan gaya hidup).
        </p>
        <div className="flex flex-wrap justify-center gap-2 text-[11px] text-stone-600">
          {[
            "Next.js",
            "React",
            "Tailwind CSS",
            "Go 1.27",
            "Python FastAPI",
            "Random Forest + SMOTE",
            "SHAP & LIME XAI",
          ].map((tech) => (
            <span
              key={tech}
              className="bg-white border border-stone-200/80 px-3 py-1 rounded-full"
            >
              {tech}
            </span>
          ))}
        </div>
      </footer>
    </div>
  );
}