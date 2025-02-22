"use client";

import { Message, useChat } from "ai/react";
import Markdown from "react-markdown";
import { useState, useRef, useEffect } from "react";
import remarkGfm from "remark-gfm";
import { ChevronDown, Pencil } from "lucide-react";

interface PlaceholderOption {
  emoji: string;
  text: string;
}

const placeholderOptions: PlaceholderOption[] = [
  { emoji: "⚡️", text: "What is Hypertension?" },
  { emoji: "🤖", text: "How do I improve my health?" },
  { emoji: "🌟", text: "What is Cardiovascular Disease?" },
];

const formatJSON = (obj: Record<string, unknown>) => {
  const processValue = (value: unknown, level: number): string => {
    const indent = "  ".repeat(level);

    if (value === null) {
      return `<span class="text-gray-400">null</span>`;
    }
    if (value === undefined) {
      return `<span class="text-red-400">undefined</span>`;
    }

    if (Array.isArray(value)) {
      if (value.length === 0) return `<span class="text-gray-300">[]</span>`;
      const items = value.map((item) => `${indent}  ${processValue(item, level + 1)}`).join(",\n");
      return `<span class="text-gray-300">[</span>\n${items}\n${indent}<span class="text-gray-300">]</span>`;
    }

    if (value && typeof value === "object") {
      const entries = Object.entries(value);
      if (entries.length === 0) return `<span class="text-gray-300">{}</span>`;

      const formattedEntries = entries
        .map(([key, val]) => {
          const formattedKey = `<span class="text-blue-400">"${key}"</span>`;
          return `${indent}  ${formattedKey}: ${processValue(val, level + 1)}`;
        })
        .join(",\n");

      return `<span class="text-gray-300">{</span>\n${formattedEntries}\n${indent}<span class="text-gray-300">}</span>`;
    }

    if (typeof value === "string") {
      const escaped = value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
      return `<span class="text-green-400">"${escaped}"</span>`;
    }

    if (typeof value === "number") {
      return `<span class="text-yellow-400">${value}</span>`;
    }

    if (typeof value === "boolean") {
      return `<span class="text-purple-400">${value}</span>`;
    }

    return `<span class="text-red-400">${String(value)}</span>`;
  };

  return `<pre class="font-mono text-sm whitespace-pre">${processValue(obj, 0)}</pre>`;
};

export default function Page() {
  const { messages, setMessages, input, handleInputChange, handleSubmit } = useChat();
  const [selectedMessage, setSelectedMessage] = useState<Message | null>(null);
  const [selectedOption, setSelectedOption] = useState<PlaceholderOption | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [showPlaceholders, setShowPlaceholders] = useState(true);

  const handleNewChat = () => {
    setMessages([]);
    setShowPlaceholders(true);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleScroll = () => {
    if (!messagesContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;
    setShouldAutoScroll(isAtBottom);
    setShowScrollButton(!isAtBottom);
  };

  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
    }
  }, [messages, shouldAutoScroll]);

  const handleOptionSelect = async (option: PlaceholderOption) => {
    setSelectedOption(option);
    const event = { target: { value: option.text } } as React.ChangeEvent<HTMLInputElement>;
    handleInputChange(event);
    setShowPlaceholders(false);
    const submitEvent = { preventDefault: () => {} } as React.FormEvent<HTMLFormElement>;
    await handleSubmit(submitEvent);
  };

  const MessageDialog = ({ message }: { message: Message }) => (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={(e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
          setSelectedMessage(null);
        }
      }}
    >
      <div className="bg-gray-900 p-6 rounded-lg max-w-3xl w-full mx-4 max-h-[80vh] overflow-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Message Details</h2>
          <button
            onClick={() => setSelectedMessage(null)}
            className="text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>
        <div
          className="bg-gray-950 p-6 rounded-lg font-mono text-sm leading-relaxed overflow-auto"
          dangerouslySetInnerHTML={{
            __html: formatJSON(message as unknown as Record<string, unknown>),
          }}
        />
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 relative scroll-smooth"
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${message.role === "user" ? "bg-gray-800" : "bg-gray-900"} p-6 rounded-lg`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center 
                ${message.role === "user" ? "bg-gray-600" : "bg-blue-600"}`}
              >
                {message.role === "user" ? "👤" : "👨🏻‍⚕️"}
              </div>
              <div className="flex-1 min-w-0">
                <div className="prose prose-invert max-w-none">
                  <div
                    className="[&>table]:w-full [&>table]:border-collapse [&>table]:my-4 
                             [&>table>thead>tr]:border-b [&>table>thead>tr]:border-gray-700 
                             [&>table>tbody>tr]:border-b [&>table>tbody>tr]:border-gray-800 
                             [&>table>*>tr>*]:p-2 [&>table>*>tr>*]:text-left 
                             [&>table>thead>tr>*]:font-semibold [&>table>tbody>tr>*]:align-top"
                  >
                    <Markdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ href, children }) => {
                          const isFootnoteRef = /^\d+$/.test(children?.[0]?.toString());
                          if (isFootnoteRef) {
                            return (
                              <a href={href} className="text-blue-400 hover:underline">
                                {children}
                              </a>
                            );
                          }
                          return (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:underline"
                            >
                              {children}
                            </a>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </Markdown>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedMessage(message)}
                  className="mt-2 text-xs text-gray-500 hover:text-gray-400"
                >
                  View Details
                </button>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="fixed bottom-24 right-8 bg-gray-700 text-white p-2 rounded-full shadow-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
        >
          <ChevronDown className="w-5 h-5" />
          <span className="pr-2">Scroll to bottom</span>
        </button>
      )}

      <div className="border-t border-gray-800 bg-gray-900 p-4">
        <div className="max-w-3xl mx-auto space-y-6">
          {showPlaceholders && messages.length === 0 && (
            <div className="flex flex-wrap gap-3 mb-4">
              {placeholderOptions.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleOptionSelect(option)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all whitespace-nowrap
                    backdrop-blur-sm bg-opacity-20 hover:bg-opacity-30
                    ${
                      selectedOption === option
                        ? "bg-blue-500/20 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.2)]"
                        : "bg-gray-800/40 text-gray-300 hover:bg-gray-700/40"
                    }`}
                >
                  <span>{option.emoji}</span>
                  <span>{option.text}</span>
                </button>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <form onSubmit={handleSubmit} className="flex gap-2 flex-1">
              <input
                value={input}
                onChange={handleInputChange}
                placeholder="Ask Laso AI..."
                className="flex-1 bg-gray-800 text-white px-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:border-gray-600 focus:ring-1 focus:ring-gray-600"
              />
              <button
                type="submit"
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </form>
            <button
              onClick={handleNewChat}
              className="bg-gray-800 text-gray-300 px-4 py-3 rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
              title="Start New Chat"
            >
              <Pencil className="w-5 h-5" />
            </button>
          </div>

          <div className="flex justify-center">
            <a
              href="https://lasoai.org"
              target="_blank"
              rel="noopener noreferrer"
              className="block transition-opacity hover:opacity-80"
            >
              <div className="flex items-center gap-2">
                <span className="text-gray-400 text-sm">Powered by</span>
                <img src="/vio-light-logo.svg" alt="Vectorize Logo" className="h-5" />
              </div>
            </a>
          </div>
        </div>
      </div>

      {selectedMessage && <MessageDialog message={selectedMessage} />}
    </div>
  );
}
