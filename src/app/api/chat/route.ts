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
Generate a detailed treatment plan for the following patient. Your answer must follow a structured format and provide clear step-by-step recommendations before introducing medications.

Output Structure:

Patient Overview: Brief summary of the patient's condition.
Lifestyle Modifications (First-Line Treatment, consider lifestyle modifications before medications)
Dietary Changes: Provide a detailed and structured diet plan using Zambian foods, including specific foods to include and avoid (e.g., DASH Diet specifics).
Physical Activity: Recommended detailed exercises with intensity and frequency.
Weight Management Strategy: Target weight, expected reduction per week, and approach.
Alcohol and Smoking: Limits and alternatives for patients struggling to quit.
Pharmacological Intervention (If Lifestyle Fails to Achieve Target BP)
Medication Options: Explain the first-line medications and alternatives based on patient profile (e.g., African ethnicity considerations).
Monitoring & Follow-Up Plan
BP Check Intervals: Frequency of follow-ups.
Tests Required: What are the tests required according to guidelines.
Special Considerations: Address ethnic, family history, and potential risks for progression to diabetes.

Here is the patient info where you need to generate treatment plan:

<question>
${messages[messages.length - 1].content}
</question>

Below are chunks of information that you can use to answer the question. Each chunk is preceded by a 
source identifier in the format [source=X&link=Y], where X is the source number and Y is the URL of the source:

<chunks>
${JSON.stringify(context)}
</chunks>
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
