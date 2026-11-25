package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

type ChatMessage struct {
	Role    string
	Content string
}

func getEnv(key, fallback string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return fallback
}

func extractString(v any) string {
	switch val := v.(type) {
	case string:
		return strings.TrimSpace(val)
	case map[string]any:
		for _, k := range []string{"text", "content", "message", "value"} {
			if s := extractString(val[k]); s != "" {
				return s
			}
		}
	}
	return ""
}

func extractUserMessage(body map[string]any) string {
	for _, k := range []string{"message", "prompt", "question", "query", "text", "content", "input"} {
		if s := extractString(body[k]); s != "" {
			return s
		}
	}
	for _, listKey := range []string{"messages", "history"} {
		if arr, ok := body[listKey].([]any); ok && len(arr) > 0 {
			for i := len(arr) - 1; i >= 0; i-- {
				if item, ok := arr[i].(map[string]any); ok {
					role := strings.ToLower(extractString(item["role"]))
					if role == "" || role == "user" {
						if s := extractString(item); s != "" {
							return s
						}
					}
				}
			}
		}
	}
	return ""
}

func extractBool(body map[string]any, keys ...string) (bool, bool) {
	for _, k := range keys {
		if v, exists := body[k]; exists && v != nil {
			switch val := v.(type) {
			case bool:
				return val, true
			case string:
				low := strings.ToLower(strings.TrimSpace(val))
				if low == "true" || low == "1" || low == "yes" {
					return true, true
				}
				if low == "false" || low == "0" || low == "no" {
					return false, true
				}
			case float64:
				return val != 0, true
			}
		}
	}
	return false, false
}

func extractPrediction(body map[string]any) map[string]any {
	for _, k := range []string{"active_prediction", "activePrediction", "prediction", "context", "dataset_context"} {
		if m, ok := body[k].(map[string]any); ok && len(m) > 0 {
			return m
		}
	}
	return nil
}

func extractHistory(body map[string]any) []ChatMessage {
	var out []ChatMessage
	for _, listKey := range []string{"history", "messages"} {
		arr, ok := body[listKey].([]any)
		if !ok {
			continue
		}
		for _, raw := range arr {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			text := extractString(item)
			if text == "" {
				continue
			}
			role := strings.ToLower(extractString(item["role"]))
			if role == "assistant" || role == "model" {
				role = "model"
			} else {
				role = "user"
			}
			out = append(out, ChatMessage{Role: role, Content: text})
		}
		if len(out) > 0 {
			break
		}
	}
	return out
}

func withCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		reqHeaders := r.Header.Get("Access-Control-Request-Headers")
		if reqHeaders != "" {
			w.Header().Set("Access-Control-Allow-Headers", reqHeaders)
		} else {
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Requested-With")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next(w, r)
	}
}

func buildSystemInstruction(useContext bool, predData map[string]any) string {
	if !useContext {
		return "Anda adalah Gradia AI Assistant, asisten cerdas yang ramah, objektif, dan informatif. " +
			"Jawab pertanyaan pengguna dalam Bahasa Indonesia secara jelas dan langsung pada intinya. " +
			"Jangan mengaitkan jawaban dengan dataset gangguan tidur atau prediksi medis kecuali pengguna secara eksplisit menanyakannya. " +
			"Aturan format: Tulis jawaban dalam teks biasa (plain text) yang rapi. DILARANG menggunakan simbol Markdown atau LaTeX seperti **, *, ###, ---, garis tabel |, atau tanda $. Gunakan angka (1., 2.) dan tanda strip (-) untuk daftar poin."
	}

	baseContext := "Anda adalah Gradia Analyst, asisten Explainable AI (XAI) untuk sistem klasifikasi gangguan tidur (Sleep Disorder Classification). " +
		"Sistem ini menggunakan model Random Forest Classifier dengan penyeimbangan kelas SMOTE pada dataset kesehatan tidur (1.500 responden, 12 fitur klinis: " +
		"Usia, Jenis Kelamin, Pekerjaan, Durasi Tidur, Kualitas Tidur, Tingkat Aktivitas Fisik, Tingkat Stres, Kategori BMI, Tekanan Darah Sistolik/Diastolik, Denyut Jantung, dan Langkah Harian), " +
		"serta dilengkapi interpretasi SHAP dan LIME. Jawab pertanyaan secara akademis, terstruktur, dan mudah dipahami dalam Bahasa Indonesia."

	if len(predData) > 0 {
		predJSON, err := json.Marshal(predData)
		if err == nil {
			baseContext += fmt.Sprintf("\n\nData simulasi/prediksi profil pengguna yang sedang aktif saat ini:\n%s", string(predJSON))
		}
	}

	baseContext += "\n\nAturan format: Tulis jawaban dalam teks biasa (plain text) yang rapi. DILARANG menggunakan simbol Markdown atau LaTeX seperti **, *, ###, ---, garis tabel |, atau tanda $ (tulis '< 80' atau '<= 4', jangan '$< 80$' atau '$\\le 4$'). Gunakan angka (1., 2.) dan tanda strip (-) untuk daftar poin."
	return baseContext
}

func callGeminiAPI(apiKey, systemInstruction, userMessage string, history []ChatMessage) (string, string, error) {
	models := []string{"gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"}

	if len(history) > 0 && history[len(history)-1].Role == "user" && history[len(history)-1].Content == userMessage {
		history = history[:len(history)-1]
	}

	var contents []map[string]any
	for _, msg := range history {
		contents = append(contents, map[string]any{
			"role":  msg.Role,
			"parts": []map[string]string{{"text": msg.Content}},
		})
	}

	contents = append(contents, map[string]any{
		"role":  "user",
		"parts": []map[string]string{{"text": userMessage}},
	})

	client := &http.Client{Timeout: 90 * time.Second}
	var lastErr error

	for i, model := range models {
		payload := map[string]any{
			"contents": contents,
		}
		if systemInstruction != "" {
			payload["system_instruction"] = map[string]any{
				"parts": []map[string]string{{"text": systemInstruction}},
			}
		}

		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			lastErr = err
			continue
		}

		endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", model)
		httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewBuffer(bodyBytes))
		if err != nil {
			lastErr = err
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-goog-api-key", apiKey)

		resp, err := client.Do(httpReq)
		if err != nil {
			lastErr = err
			log.Printf("[Gemini Error] Gagal terhubung ke %s: %v", model, err)
			continue
		}

		respBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBytes))
			log.Printf("[Gemini Error] Model %s ditolak (%d): %s", model, resp.StatusCode, string(respBytes))
			if (resp.StatusCode == http.StatusServiceUnavailable || resp.StatusCode == http.StatusTooManyRequests) && i < len(models)-1 {
				time.Sleep(1500 * time.Millisecond)
			}
			continue
		}

		var parsed struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal(respBytes, &parsed); err != nil {
			lastErr = err
			continue
		}

		if len(parsed.Candidates) > 0 && len(parsed.Candidates[0].Content.Parts) > 0 {
			replyText := strings.TrimSpace(parsed.Candidates[0].Content.Parts[0].Text)
			if replyText != "" {
				log.Printf("[Gemini OK] Berhasil dijawab oleh %s", model)
				return replyText, model, nil
			}
		}
	}

	return "", "", lastErr
}

func buildGroundedFallback(useContext bool, predData map[string]any, reason string) string {
	if !useContext {
		return "Mode Konteks Dataset sedang nonaktif, sehingga atribut penelitian tidak disertakan. " +
			"Saat ini server Go berjalan pada mode fallback lokal (" + reason + ")."
	}

	pred := "Insomnia"
	conf := "61.45"
	if predData != nil {
		if v, ok := predData["prediction"]; ok && fmt.Sprintf("%v", v) != "" {
			pred = fmt.Sprintf("%v", v)
		}
		if v, ok := predData["confidence_percent"]; ok && fmt.Sprintf("%v", v) != "" {
			conf = fmt.Sprintf("%v", v)
		}
	}

	return fmt.Sprintf(
		"1. Ringkasan Klasifikasi: Model Random Forest Classifier dengan penyeimbangan SMOTE pada data latih mengklasifikasikan profil simulasi saat ini ke dalam kategori %s dengan tingkat keyakinan %s%%.\n"+
			"2. Determinan Utama Model: Variabel yang memberikan kontribusi paling signifikan terhadap pemisahan kelas pada dataset meliputi Tekanan Darah Sistolik, Kategori BMI, Durasi Tidur harian, dan Tingkat Stres.\n"+
			"3. Status Layanan LLM: Sistem sedang menggunakan mode fallback lokal (%s).",
		pred, conf, reason,
	)
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	rawBody, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Gagal membaca request", http.StatusBadRequest)
		return
	}
	log.Printf("[Chat Incoming] %s", string(rawBody))

	var body map[string]any
	if err := json.Unmarshal(rawBody, &body); err != nil {
		http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
		return
	}

	userMsg := extractUserMessage(body)
	if userMsg == "" {
		http.Error(w, "Pesan tidak boleh kosong", http.StatusBadRequest)
		return
	}

	useContext, _ := extractBool(body, "use_context", "useContext", "include_context", "includeContext", "grounded")
	predData := extractPrediction(body)
	history := extractHistory(body)
	systemPrompt := buildSystemInstruction(useContext, predData)

	apiKey := getEnv("GEMINI_API_KEY", "")

	provider := "Gateway Fallback (Lokal)"
	modelUsed := "gradia-grounded-context"
	var reply string

	if apiKey != "" {
		geminiReply, activeModel, err := callGeminiAPI(apiKey, systemPrompt, userMsg, history)
		if err == nil && geminiReply != "" {
			reply = geminiReply
			provider = "Google Gemini API"
			modelUsed = activeModel
		} else {
			reply = buildGroundedFallback(useContext, predData, "koneksi ke Gemini API gagal")
		}
	} else {
		reply = buildGroundedFallback(useContext, predData, "GEMINI_API_KEY belum diatur")
	}

	wantsSSE := false
	if streamVal, exists := extractBool(body, "stream"); exists {
		wantsSSE = streamVal
	}

	if wantsSSE {
		if flusher, ok := w.(http.Flusher); ok {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			words := strings.SplitAfter(reply, " ")
			for _, chunk := range words {
				eventPayload, _ := json.Marshal(map[string]any{
					"delta":    chunk,
					"token":    chunk,
					"chunk":    chunk,
					"text":     chunk,
					"content":  chunk,
					"provider": provider,
					"model":    modelUsed,
					"grounded": useContext,
					"done":     false,
				})
				fmt.Fprintf(w, "data: %s\n\n", eventPayload)
				flusher.Flush()
				time.Sleep(10 * time.Millisecond)
			}

			donePayload, _ := json.Marshal(map[string]any{
				"delta":    "",
				"token":    "",
				"chunk":    "",
				"text":     "",
				"content":  "",
				"reply":    reply,
				"response": reply,
				"answer":   reply,
				"provider": provider,
				"model":    modelUsed,
				"grounded": useContext,
				"done":     true,
			})
			fmt.Fprintf(w, "data: %s\n\n", donePayload)
			flusher.Flush()
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"reply":    reply,
		"response": reply,
		"answer":   reply,
		"text":     reply,
		"content":  reply,
		"provider": provider,
		"model":    modelUsed,
		"grounded": useContext,
	})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	apiKey := getEnv("GEMINI_API_KEY", "")
	apiKeyConfigured := apiKey != ""

	provider := "Gateway Fallback (Lokal)"
	model := "gradia-grounded-context"
	if apiKeyConfigured {
		provider = "Google Gemini API"
		model = "gemini-3.8-flash"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":         "ok",
		"gateway":        "Gradia Go Gateway",
		"port":           getEnv("PORT", "8080"),
		"route":          "jalur A",
		"llm_configured": apiKeyConfigured,
		"provider":       provider,
		"model":          model,
	})
}

func main() {
	mlBaseURL := getEnv("ML_API_URL", "http://127.0.0.1:8000")
	mlTarget, err := url.Parse(mlBaseURL)
	if err != nil {
		log.Fatalf("Gagal parsing ML_API_URL (%s): %v", mlBaseURL, err)
	}

	mlProxy := httputil.NewSingleHostReverseProxy(mlTarget)
	originalDirector := mlProxy.Director
	mlProxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = mlTarget.Host
		req.Header.Set("X-Forwarded-Host", mlTarget.Host)
	}
	mlProxy.ModifyResponse = func(resp *http.Response) error {
		resp.Header.Del("Access-Control-Allow-Origin")
		resp.Header.Del("Access-Control-Allow-Methods")
		resp.Header.Del("Access-Control-Allow-Headers")
		return nil
	}
	mlProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("[ML Proxy Error] Gagal meneruskan ke %s: %v", mlTarget.String(), err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Layanan ML tidak dapat dijangkau (%s): %v", mlBaseURL, err),
		})
	}

	http.HandleFunc("/api/chat", withCORS(handleChat))
	http.HandleFunc("/chat", withCORS(handleChat))
	http.HandleFunc("/api/health", withCORS(handleHealth))
	http.HandleFunc("/health", withCORS(handleHealth))

	http.HandleFunc("/", withCORS(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			r.URL.Path = strings.TrimPrefix(r.URL.Path, "/api")
		}
		mlProxy.ServeHTTP(w, r)
	}))

	port := getEnv("PORT", "8080")
	addr := ":" + port
	log.Printf("Gradia Go Gateway aktif pada port %s -> meneruskan ML ke %s", port, mlBaseURL)

	if getEnv("GEMINI_API_KEY", "") != "" {
		log.Printf("Status LLM: GEMINI_API_KEY terdeteksi (Google Gemini API aktif)")
	} else {
		log.Printf("Status LLM: GEMINI_API_KEY belum diatur (menggunakan Gateway Fallback Lokal)")
	}

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server Go berhenti: %v", err)
	}
}