FROM golang:1.27-alpine AS builder

WORKDIR /app

COPY backend-go/go.mod ./
RUN go mod download

COPY backend-go/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

FROM alpine:3.20

RUN apk --no-cache add ca-certificates tzdata
WORKDIR /app

COPY --from=builder /app/server ./server

EXPOSE 8080
CMD ["./server"]
