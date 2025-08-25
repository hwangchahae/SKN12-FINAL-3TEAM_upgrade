"use client";

import { useEffect, useState } from "react";

interface TextTypeProps {
  text: string;
  speed?: number;
  className?: string;
  showCursor?: boolean;
  cursorCharacter?: string;
}

const TextType = ({ 
  text, 
  speed = 100, 
  className = "",
  showCursor = true,
  cursorCharacter = "|"
}: TextTypeProps) => {
  const [displayText, setDisplayText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  console.log("🚀 TextType STARTED with text:", text);
  console.log("🚀 Current displayText:", displayText);
  console.log("🚀 Current index:", currentIndex);

  // 한글 문자열을 올바르게 분할하는 함수
  const splitKoreanText = (text: string) => {
    const result = Array.from(text);
    console.log("🚀 Split result:", result);
    return result;
  };

  const textArray = splitKoreanText(text);

  // 타이핑 효과
  useEffect(() => {
    console.log("🚀 useEffect triggered - currentIndex:", currentIndex, "textArray length:", textArray.length);
    
    if (currentIndex < textArray.length) {
      console.log("🚀 Setting timer for character:", textArray[currentIndex]);
      
      const timer = setTimeout(() => {
        const newText = displayText + textArray[currentIndex];
        console.log("🚀 Adding character:", textArray[currentIndex], "New text:", newText);
        
        setDisplayText(newText);
        setCurrentIndex(currentIndex + 1);
      }, speed);

      return () => {
        console.log("🚀 Clearing timer");
        clearTimeout(timer);
      };
    } else {
      console.log("🚀 Typing complete!");
    }
  }, [currentIndex, textArray, speed, displayText]);

  console.log("🚀 Rendering with displayText:", displayText);

  return (
    <span className={className}>
      {displayText}
      {showCursor && currentIndex < textArray.length && (
        <span className="ml-1 text-white animate-pulse">{cursorCharacter}</span>
      )}
    </span>
  );
};

export default TextType; 