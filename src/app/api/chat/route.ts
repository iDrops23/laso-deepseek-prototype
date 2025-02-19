import { CoreMessage, streamText } from "ai";
import { groq } from "@ai-sdk/groq";

export async function POST(req: Request) {
  const { messages }: { messages: CoreMessage[] } = await req.json();
  const model = process.env.MODEL;

  if (!model) {
    throw new Error(
      "MODEL is not set in the environment variables. Please add it to your environment settings (e.g. .env.develoment) file.",
    );
  }
  let context = null;

  if (messages.length > 0) {
    const previousMessages = messages.slice(0, -1);
    context = await retrieveData(String(messages[messages.length - 1].content), previousMessages);
  } else {
    console.log("No messages available, skipping retrieveData.");
  }

  // Override the last message to include the context and instructions
  if (messages.length > 0 && context) {
    messages[messages.length - 1].content = `
You are tasked with generating a treatment plan using provided chunks of information, consider first steps first before using medications. Your goal is to provide an accurate answer while citing your sources using a specific markdown format.

Here is the question you need to answer:
<question>
${messages[messages.length - 1].content}
</question>

Below are chunks of information that you can use to answer the question. Each chunk is preceded by a 
source identifier in the format [source=X&link=Y], where X is the source number and Y is the URL of the source:

<chunks>
${JSON.stringify(context)}
</chunks>

Your task is to answer the question using the information provided in these chunks. 
When you use information from a specific chunk in your answer, you must cite it using a markdown link format. 
The citation should appear at the end of the sentence where the information is used.

If you cannot answer the question using the provided chunks, say "Sorry I don't know".

The citation format should be as follows:
[Chunk source](URL)

For example, if you're using information from the chunk labeled [source=3&link=https://example.com/page], your citation would look like this:
[3](https://example.com/page) and would open a new tab to the source URL when clicked.
`;
  }

  console.log("sending messages to the LLM", messages);

  const result = await streamText({
    model: groq(model) as any,
    system:
      "You are the best Hypertension Doctor capable of generating the best treatment plan based on the patient overview. You always generate your responses as correctly structured, valid markdown.",
    messages,
  });

  return result.toDataStreamResponse();
}

const retrieveData = async (question: string, messages: CoreMessage[]) => {
  const token = process.env.VECTORIZE_TOKEN;
  const pipelineRetrievalUrl = process.env.VECTORIZE_RETRIEVAL_URL;
  const groqApiKey = process.env.GROQ_API_KEY;

  if (!token) {
    throw new Error(
      "VECTORIZE_TOKEN is not set in the environment variables. Please add it to your environment settings (e.g. .env.develoment) file.",
    );
  }

  if (!pipelineRetrievalUrl) {
    throw new Error(
      "pipelineRetrievalUrl is not set. Please define the URL in the environment settings (e.g. in the file .env.development) before calling retrieveData.",
    );
  }

  if (!groqApiKey) {
    throw new Error(
      "GROQ_API_KEY is not set in the environment variables. Please add it to your environment settings (e.g. .env.develoment) file.",
    );
  }

  const payload: any = {
    question,
    numResults: 5,
    rerank: true,
  };

  if (messages && messages.length > 0) {
    payload.context = {
      messages: messages,
    };
  }

  const response = await fetch(pipelineRetrievalUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: token,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch data: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
};
