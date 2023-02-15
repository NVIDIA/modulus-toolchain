<template>
    <div>
        <v-card>
            <v-row>
                <v-col>
                    <h2>Neural Network Config</h2>
                </v-col>

                <v-col align="right" cols="auto">
                    <v-btn @click="save_config">save</v-btn>
                </v-col>
            </v-row>
            <div>
                <pdcontainer></pdcontainer>
            </div>
        </v-card>
    </div>
</template>