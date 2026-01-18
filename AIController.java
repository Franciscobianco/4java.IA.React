
Java
package com.example.ia;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/ia")
public class AIController {

    private final AIService aiService;

    public AIController(AIService aiService) {
        this.aiService = aiService;
    }

    @PostMapping("/clasificar")
    public String clasificar(@RequestBody String texto) {
        return aiService.clasificarTexto(texto);
    }
}


