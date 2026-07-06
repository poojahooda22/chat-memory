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