/** Typed client for the chat-memory backend (the four Phase 1/2 endpoints). */
import axios from "axios";

import { BACKEND_URL, USER_ID } from "./config";

const http = axios.create({ baseURL: BACKEND_URL });

export interface Memory {
  id: string;
  user_id: string;
  content: string;
  source_episode_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface HistoryEntry {
  event: "ADD" | "UPDATE" | "DELETE";
  old_content: string | null;
  new_content: string | null;
  created_at: string;
}

export interface MemoryOperation {
  event: string;
  memory_id: string | null;
  text: string;
}

export interface ChatResponse {
  reply: string;
  memories_used: string[];
  photos_used: string[];
  operations: MemoryOperation[];
}

export async function sendChat(
  message: string,
  conversationId: string,
): Promise<ChatResponse> {
  const { data } = await http.post<ChatResponse>("/chat", {
    user_id: USER_ID,
    conversation_id: conversationId,
    message,
  });
  return data;
}

export async function listMemories(): Promise<Memory[]> {
  const { data } = await http.get<Memory[]>("/memories", { params: { user_id: USER_ID } });
  return data;
}

export async function getMemoryHistory(memoryId: string): Promise<HistoryEntry[]> {
  const { data } = await http.get<HistoryEntry[]>(`/memories/${memoryId}/history`);
  return data;
}

export async function deleteMemory(memoryId: string): Promise<void> {
  await http.delete(`/memories/${memoryId}`);
}

// ── image ingest ─────────────────────────────────────────────────────────────

export interface EntityChip {
  index: number;
  type: "person" | "pet" | "object";
  description: string;
  confidence: number | null;
  label: string | null;
  /** who labeled it: "user" (you) or "memory" (visual recognition) */
  labeled_by: "user" | "memory" | null;
  /** recognition: looks like an entity you already named — confirm to apply */
  suggested_name: string | null;
}

export interface IngestJob {
  id: string;
  kind: "photo" | "screenshot";
  status: "queued" | "processing" | "done" | "failed";
  filename: string;
  captured_at: string | null;
  time_source: string | null;
  place: string | null; // geocoded place name, when the photo carried GPS
  episode_id: string | null;
  caption: string | null;
  entities: EntityChip[];
  error: string | null;
  created_at: string;
}

/** EXIF contract: append the RAW File objects — never a canvas re-encode, which would strip
 * the photo's capture time + GPS before the bytes ever leave the browser. */
export async function uploadImages(files: File[]): Promise<IngestJob[]> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("user_id", USER_ID);
  const { data } = await http.post<IngestJob[]>("/uploads", form);
  return data;
}

export async function listUploads(): Promise<IngestJob[]> {
  const { data } = await http.get<IngestJob[]>("/uploads", { params: { user_id: USER_ID } });
  return data;
}

export async function retryUpload(jobId: string): Promise<void> {
  await http.post(`/uploads/${jobId}/retry`);
}

export async function renameUpload(jobId: string, filename: string): Promise<void> {
  await http.patch(`/uploads/${jobId}`, { filename });
}

/** Forget this photo: file + episode + links removed; single-source memories forgotten. */
export async function deleteUpload(jobId: string): Promise<void> {
  await http.delete(`/uploads/${jobId}`);
}

export function uploadImageUrl(jobId: string): string {
  return `${BACKEND_URL}/uploads/${jobId}/image`;
}

// ── the relationship graph ───────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  name: string;
  type: "person" | "pet" | "object";
  photo_count: number;
  representative_job_id: string | null;
  photo_job_ids: string[]; // every photo this entity is in — the membership spokes
}

export interface GraphEdge {
  src: string;
  dst: string;
  weight: number;
  cooccur_count: number;
  is_learning: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export async function getGraph(): Promise<GraphData> {
  const { data } = await http.get<GraphData>("/graph", { params: { user_id: USER_ID } });
  return data;
}

export interface LabelResponse {
  entity: { id: string; name: string; type: string; description: string };
  memory_event: string;
  reused_existing: boolean;
}

/** Detach a label from a photo — the undo for a wrong auto-recognition. */
export async function unlabelEntity(episodeId: string, entityIndex: number): Promise<void> {
  await http.delete(`/episodes/${episodeId}/label/${entityIndex}`);
}

/** Name a detected entity on a photo episode: "this is Monty". */
export async function labelEntity(
  episodeId: string,
  entityIndex: number,
  name: string,
): Promise<LabelResponse> {
  const { data } = await http.post<LabelResponse>(`/episodes/${episodeId}/label`, {
    entity_index: entityIndex,
    name,
  });
  return data;
}